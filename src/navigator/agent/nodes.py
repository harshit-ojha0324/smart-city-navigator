"""
Shared node helpers and the `understand` entry node.

The plan/tools/finish loop now lives per specialist agent in workers.py; the
supervisor topology lives in graph.py. This module holds what both reuse: the
stream-writer emit helper, message parsing, the query-understanding node, and
the deterministic answer composer.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from navigator.core.geocode import display_name, rider_lines

from .llm import DeterministicPlanner, prompt_suffix


def message_text(content) -> str:
    """Plain text of a message's content. Gemini can return a list of content
    parts rather than a string; downstream code (synthesis, SSE) wants text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p if isinstance(p, str) else str(p.get("text", ""))
                       for p in content if isinstance(p, (str, dict)))
    return str(content or "")


def last_ai(messages: list):
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def tool_results(messages: list) -> list[tuple[str, object]]:
    """(tool_name, parsed_payload) for every ToolMessage, newest last."""
    out = []
    for m in messages:
        if isinstance(m, ToolMessage):
            payload = m.content
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    pass
            out.append((getattr(m, "name", "") or "", payload))
    return out


# ── understand (supervisor entry) ──────────────────────────────────────
ROUTER_PROMPT = """You route questions for a New York City subway assistant to \
specialist agents. Classify the rider's question.

intent — exactly one of:
  route   planning a trip between two places, or asking to get somewhere
  status  service status, delays, suspensions, elevator/escalator outages
  info    which lines serve a station, nearest station to a place
  other   anything the subway assistant cannot answer

needs_status — true only when a trip question ALSO asks about delays or service.

Reply with JSON only, no prose: {"intent": "...", "needs_status": true|false}"""

_JSON_OBJECT = re.compile(r"\{.*?\}", re.S)
_INTENTS = {"route", "status", "info", "other"}


async def classify_with_llm(llm, question: str) -> tuple[str, bool] | None:
    """Ask the model to route. Returns None on anything unusable, so the caller
    can fall back to the regex planner rather than fail the turn."""
    try:
        reply = await llm.ainvoke([SystemMessage(content=ROUTER_PROMPT + prompt_suffix()),
                                   HumanMessage(content=question)])
        match = _JSON_OBJECT.search(message_text(reply.content))
        if not match:
            return None
        parsed = json.loads(match.group(0))
        intent = str(parsed.get("intent", "")).strip().lower()
        if intent not in _INTENTS:
            return None
        return intent, bool(parsed.get("needs_status", False))
    except Exception:
        return None


def make_understand_node(planner: DeterministicPlanner, llm=None):
    """Entry node: decide what the rider is asking for.

    With a model configured the LLM routes; the regex planner is the fallback
    for a malformed reply, a refusal, or an unreachable endpoint — a flaky
    model degrades the routing, it never breaks the turn.
    """
    async def understand(state) -> dict:
        question = state["question"]
        routed = await classify_with_llm(llm, question) if llm is not None else None
        if routed is None:
            intent, needs_status = planner.classify(question), None
            router = "planner"
        else:
            intent, needs_status = routed
            router = "llm"
        step = {"node": "understand", "intent": intent, "router": router,
                "text": f"Understood the question; intent = '{intent}' (routed by {router})."}
        # Every turn's question joins the thread's conversation history.
        return {"intent": intent, "needs_status": needs_status, "router": router,
                "steps": [step], "messages": [HumanMessage(content=question)]}

    return understand


# ── deterministic answer composer (shared by workers + synthesis) ──────
def compose_answer(intent: str, results: list[tuple[str, object]], question: str = "") -> str:
    """Grounded natural-language answer from a set of tool results."""
    if not results:
        if intent == "other":
            return ("I'm a NYC subway trip planner — I can help with routes between "
                    "stations, live service status, and finding the nearest station. "
                    "What trip can I plan for you?")
        if intent == "route":
            dest = DeterministicPlanner().destination_only(question)
            where = f" to {dest}" if dest else ""
            return (f"Where are you starting from? Tell me your origin station or "
                    f"neighborhood and I'll plan the trip{where}.")
        return "I couldn't find a tool to answer that. Try asking about a subway route or line status."

    # Several *distinct* line checks (the Service Advisor was handed the lines of
    # a planned route) → one per-line report scoped to the rider's trip. A model
    # that asked about the same line repeatedly gets one answer, not three.
    by_line = {p.get("line"): p for n, p in results
               if n == "get_line_status" and isinstance(p, dict)}
    line_checks = list(by_line.values())
    if len(line_checks) > 1:
        parts = []
        for p in line_checks:
            if p.get("error"):
                continue
            detail = "" if p.get("severity", 0) == 0 else f" — {str(p.get('message', '')).rstrip('.')}"
            parts.append(f"{p.get('line')}: {p.get('status')}{detail}")
        note = _simulated_note(line_checks)
        return "Service on your route — " + "; ".join(parts) + "." + note

    name, payload = results[-1]
    if not isinstance(payload, dict):
        # A tool that raised comes back as an error string, not a payload. Say so
        # instead of narrating success over a tool that never produced data.
        return ("I hit an error looking that up and don't want to guess. "
                "Try rephrasing, or ask about a different station or line.")
    p = payload

    if name in {"plan_trip", "plan_trip_by_id"}:
        if p.get("needs_disambiguation"):
            bits = []
            for label in ("origin", "destination"):
                side = p.get(label, {})
                if side.get("ambiguous"):
                    opts = " or ".join(_choice(c) for c in side.get("candidates", []))
                    bits.append(f"the {label} '{side.get('name')}' could be: {opts}")
            return "I need a bit more detail — " + "; ".join(bits) + ". Which did you mean?"
        if p.get("error") or not p.get("found", True):
            return f"I couldn't plan that trip: {p.get('error', 'no route found')}."
        return p.get("summary", "Route planned.")

    if name == "get_service_status":
        return p.get("summary", "Service status unavailable right now.")

    if name == "get_line_status":
        if p.get("error"):
            return p["error"]
        return (f"The {p.get('line')} train: {p.get('status')} "
                f"(severity {p.get('severity')}/3). {p.get('message', '')}").strip() + _simulated_note([p])

    if name == "resolve_station":
        if p.get("error"):
            return p["error"]
        s = p.get("station", {})
        lines = rider_lines(s.get("complex_lines") or s.get("lines"))
        base = (f"{s.get('name')} is served by the {', '.join(lines)} "
                f"{'trains' if len(lines) != 1 else 'train'}.")
        others = [_choice(c) for c in p.get("candidates", []) if c.get("id") != s.get("id")]
        if p.get("ambiguous") and others:
            base += (f" Heads up — several stations share that name; I answered for the "
                     f"{'/'.join(lines)} stop. The others: {', '.join(others)}.")
        return base

    if name == "geocode_place":
        if p.get("error"):
            return p["error"]
        near = p.get("nearest_station", {})
        matched = str(p.get("matched", ""))
        if p.get("source") == "landmark":
            matched = display_name(matched)  # station names keep their own casing
        return (f"{matched} is near {near.get('name')} "
                f"({near.get('distance_km')} km away).")

    if name == "nearest_station":
        return f"The nearest station is {p.get('name')} ({p.get('distance_km')} km away)."

    if name == "list_elevator_outages":
        if p.get("available") is False:
            return p.get("message", "Elevator/escalator outage data is unavailable right now.")
        where = f" at {p['station']}" if p.get("station") else ""
        n = p.get("count", 0)
        if n == 0:
            return f"No elevator or escalator outages are currently reported{where}."
        shown = "; ".join(
            f"{o.get('type')} at {o.get('station')} ({o.get('serving') or 'n/a'}) — "
            f"{o.get('reason') or 'out of service'}, back {o.get('eta') or 'TBD'}"
            for o in p.get("outages", [])[:3])
        more = f" (+{n - 3} more)" if n > 3 else ""
        return f"{n} elevator/escalator outage{'s' if n != 1 else ''}{where}: {shown}{more}."

    if name == "list_stations":
        count = p.get("count", 0)
        names = ", ".join(s["name"] for s in p.get("stations", [])[:8])
        more = " …" if p.get("truncated") or count > 8 else ""
        return f"{count} stations match{': ' + names + more if names else ''}."

    return json.dumps(payload)[:500]


def _choice(candidate: dict) -> str:
    """"Fulton St (J/Z)" — the lines are what tell two same-named stations apart."""
    lines = rider_lines(candidate.get("lines"))
    return f"{candidate['name']} ({'/'.join(lines)})" if lines else candidate["name"]


def _simulated_note(line_payloads: list[dict]) -> str:
    if any(p.get("source") == "simulated" for p in line_payloads):
        return " (Live MTA feed unavailable — this status is simulated.)"
    return ""
