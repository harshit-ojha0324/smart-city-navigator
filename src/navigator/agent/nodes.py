"""
Shared node helpers and the `understand` entry node.

The plan/tools/finish loop now lives per specialist agent in workers.py; the
supervisor topology lives in graph.py. This module holds what both reuse: the
stream-writer emit helper, message parsing, the query-understanding node, and
the deterministic answer composer.
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import StreamWriter

from .llm import DeterministicPlanner


def _emit(step: dict, writer: StreamWriter | None) -> None:
    """Stream one reasoning step. On Python 3.10 the writer must be injected into
    the node signature (get_stream_writer doesn't cross async tasks pre-3.11)."""
    if writer is None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            writer = None
    if writer is not None:
        try:
            writer(step)
        except Exception:
            pass


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
def make_understand_node(planner: DeterministicPlanner):
    async def understand(state, writer: StreamWriter = None) -> dict:
        question = state["question"]
        intent = planner.classify(question)
        step = {"node": "understand", "intent": intent,
                "text": f"Understood the question; intent = '{intent}'."}
        _emit(step, writer)
        updates: dict = {"intent": intent, "steps": [step]}
        if not state.get("messages"):
            updates["messages"] = [HumanMessage(content=question)]
        return updates

    return understand


# ── deterministic answer composer (shared by workers + synthesis) ──────
def compose_answer(intent: str, results: list[tuple[str, object]], question: str = "") -> str:
    """Grounded natural-language answer from a set of tool results."""
    if not results:
        if intent == "other":
            return ("I'm a NYC subway trip planner — I can help with routes between "
                    "stations, live service status, and finding the nearest station. "
                    "What trip can I plan for you?")
        return "I couldn't find a tool to answer that. Try asking about a subway route or line status."

    name, payload = results[-1]
    p = payload if isinstance(payload, dict) else {}

    if name in {"plan_trip", "plan_trip_by_id"}:
        if p.get("needs_disambiguation"):
            bits = []
            for label in ("origin", "destination"):
                side = p.get(label, {})
                if side.get("ambiguous"):
                    opts = ", ".join(c["name"] for c in side.get("candidates", []))
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
                f"(severity {p.get('severity')}/3). {p.get('message', '')}").strip()

    if name == "resolve_station":
        if p.get("error"):
            return p["error"]
        s = p.get("station", {})
        base = f"{s.get('name')} is served by the {', '.join(s.get('lines', []))} line(s)."
        if p.get("ambiguous"):
            opts = ", ".join(c["name"] for c in p.get("candidates", []))
            base += f" Note: that name is ambiguous — also matches {opts}."
        return base

    if name == "geocode_place":
        if p.get("error"):
            return p["error"]
        near = p.get("nearest_station", {})
        return (f"{p.get('matched')} is near {near.get('name')} "
                f"({near.get('distance_km')} km away).")

    if name == "nearest_station":
        return f"The nearest station is {p.get('name')} ({p.get('distance_km')} km away)."

    if name == "list_elevator_outages":
        n = p.get("count", 0)
        return f"There {'is' if n == 1 else 'are'} currently {n} elevator/escalator outage(s) matching."

    if name == "list_stations":
        return f"{p.get('count', 0)} stations match."

    return json.dumps(payload)[:500]
