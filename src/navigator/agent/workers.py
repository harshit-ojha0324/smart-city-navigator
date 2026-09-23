"""
Specialist worker agents.

Each worker is an *independent* LangGraph agent — its own compiled StateGraph with
a plan → tools → finish loop, its own persona/system prompt, and its own slice of
the toolset. The supervisor (graph.py) delegates a task to one or more of them and
collects their results.

    RoutePlanner   — trip planning + geocoding tools
    ServiceAdvisor — live service status + outage tools
    StationInfo    — station lookup + nearest-station tools

With a Gemini key each worker's `plan` node is the LLM doing tool-calling under
that worker's persona; offline, a per-worker deterministic policy emits the same
calls. Either way each worker runs its own loop, so this is genuinely multi-agent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter

from navigator.core.graph_data import feed_line

from .llm import DeterministicPlanner, prompt_suffix
from .nodes import _emit, compose_answer, last_ai, message_text, tool_results
from .state import WorkerState

# A model that keeps calling tools instead of answering will loop until the
# graph's recursion limit and take the whole turn down with it — seen with a
# local 8B model. After this many tool rounds the agent stops asking and
# composes its answer from what the tools already returned.
MAX_TOOL_ROUNDS = int(os.environ.get("NAVIGATOR_MAX_TOOL_ROUNDS", "3"))


@dataclass
class WorkerSpec:
    name: str
    persona: str
    tool_names: tuple[str, ...]
    intent: str  # intent label the deterministic policy plans under


# Appended to every persona: the difference between an agent that reports tool
# output and one that improvises over it.
GROUNDING = (
    " Call a tool before answering — never answer from memory. Base every fact on the "
    "tool output: when it returns a `summary`, relay that summary faithfully (light "
    "rephrasing is fine, but never change a line, station, time or count). If the tool "
    "returns an error, asks for disambiguation, or gives nothing useful, say so plainly "
    "and ask for what you need — never invent a route, station or service status. "
    "Answer in at most three sentences, with no preamble."
)


WORKER_SPECS: list[WorkerSpec] = [
    WorkerSpec(
        name="route_planner",
        persona=("You are the Route Planner agent. You own trip planning: resolve the "
                 "origin and destination, then plan the fastest subway route and describe "
                 "the legs and transfers. Never invent a route."),
        tool_names=("plan_trip", "plan_trip_by_id", "geocode_place", "resolve_station", "list_stations"),
        intent="route",
    ),
    WorkerSpec(
        name="service_advisor",
        persona=("You are the Service Advisor agent. You own live service status: report "
                 "delays, suspensions, planned work, and elevator/escalator outages from "
                 "the real-time feed. Ground every claim in the tool output."),
        tool_names=("get_service_status", "get_line_status", "list_elevator_outages"),
        intent="status",
    ),
    WorkerSpec(
        name="station_info",
        persona=("You are the Station Info agent. You answer which lines serve a station "
                 "and find the nearest station to a place or coordinate."),
        tool_names=("resolve_station", "nearest_station", "geocode_place", "list_stations"),
        intent="info",
    ),
]


def _select(tools: list, names: tuple[str, ...]) -> list:
    by_name = {t.name: t for t in tools}
    return [by_name[n] for n in names if n in by_name]


def _carry_forward(results: list[tuple[str, object]]) -> dict:
    """The payload worth passing to the next agent: the itinerary if one was
    produced, else the last structured result."""
    dicts = [payload for _name, payload in results if isinstance(payload, dict)]
    for payload in reversed(dicts):
        if payload.get("legs"):
            return payload
    return dicts[-1] if dicts else {}


def _route_worker_plan(state) -> str:
    last = last_ai(state["messages"])
    return "tools" if (last is not None and getattr(last, "tool_calls", None)) else "finish"


def build_worker(spec: WorkerSpec, tools: list, llm, planner: DeterministicPlanner):
    """Compile one specialist agent as a standalone plan → tools → finish subgraph."""
    my_tools = _select(tools, spec.tool_names)
    llm_with_tools = llm.bind_tools(my_tools) if llm is not None else None

    async def plan(state) -> dict:
        messages = state["messages"]
        results_in = any(isinstance(m, ToolMessage) for m in messages)
        rounds = sum(1 for m in messages
                     if isinstance(m, AIMessage) and getattr(m, "tool_calls", None))
        if rounds >= MAX_TOOL_ROUNDS:
            # Out of patience: an empty reply routes straight to `finish`, which
            # answers from the tool results already in hand.
            return {"messages": [AIMessage(content="")]}
        if llm_with_tools is not None:
            system = SystemMessage(content=spec.persona + GROUNDING + prompt_suffix())
            ai = await llm_with_tools.ainvoke([system] + messages)
            return {"messages": [ai]}
        if not results_in:
            calls = planner.initial_tool_calls(state["task"], spec.intent,
                                               state.get("context_lines") or None)
            return {"messages": [AIMessage(content="", tool_calls=calls)]}
        return {"messages": [AIMessage(content="")]}

    async def finish(state) -> dict:
        messages = state["messages"]
        last = last_ai(messages)
        results = tool_results(messages)
        # Structured data is kept on both paths so the supervisor can hand it to
        # the next agent (route lines → service status). An itinerary wins over a
        # later lookup: a model that plans a trip and then checks a station name
        # must still hand the route on.
        data = _carry_forward(results)
        text = message_text(last.content) if last is not None else ""
        if llm is not None and text and not getattr(last, "tool_calls", None):
            result = text
        else:
            result = compose_answer(spec.intent, results, state["task"])
        return {"result": result, "data": data if isinstance(data, dict) else {}}

    builder = StateGraph(WorkerState)
    builder.add_node("plan", plan)
    builder.add_node("tools", ToolNode(my_tools))
    builder.add_node("finish", finish)
    builder.set_entry_point("plan")
    builder.add_conditional_edges("plan", _route_worker_plan, {"tools": "tools", "finish": "finish"})
    builder.add_edge("tools", "plan")
    builder.add_edge("finish", END)
    return builder.compile()


def _route_lines(worker_results: list[dict]) -> list[str]:
    """Lines ridden on a route an earlier agent planned, in trip order.

    Mapped to the ids the alerts feed publishes: a leg on the "7X" is the 7
    train as far as service status is concerned, and asking the feed about "7X"
    would come back "unknown line" — dropping that leg from the report.
    """
    for w in worker_results:
        if w.get("agent") == "route_planner":
            legs = (w.get("data") or {}).get("legs") or []
            ridden = [leg["line"] for leg in legs
                      if not leg.get("walk") and leg.get("line") not in (None, "?")]
            return list(dict.fromkeys(feed_line(line) for line in ridden))
    return []


def make_worker_node(spec: WorkerSpec, subgraph):
    """A supervisor node that delegates the task to one worker agent's subgraph."""
    async def node(state, writer: StreamWriter = None) -> dict:
        lines = _route_lines(state.get("worker_results", [])) if spec.name == "service_advisor" else []

        handoff = f" Hand-off: route uses lines {', '.join(lines)}." if lines else ""
        _emit({"node": spec.name, "text": f"Supervisor → {spec.name}: delegating the task.{handoff}"}, writer)
        prompt = state["question"]
        if lines:  # the LLM path gets the same hand-off as the deterministic one
            prompt += (f"\n\n[From the Route Planner agent: the planned route rides the "
                       f"{', '.join(lines)} line(s). Report service status for exactly those lines.]")
        try:
            out = await subgraph.ainvoke({
                "task": state["question"],
                "messages": [HumanMessage(content=prompt)],
                "context_lines": lines,
                "result": "", "data": {},
            })
        except Exception as exc:  # one agent failing must not take down the turn
            _emit({"node": spec.name, "text": f"{spec.name} failed: {type(exc).__name__}"}, writer)
            return {
                "worker_results": [{"agent": spec.name, "data": {}, "result":
                                    "I couldn't complete that lookup — try asking again, "
                                    "or rephrase the station or line."}],
                "steps": [{"node": spec.name, "text": f"failed: {type(exc).__name__}"}],
            }
        result = out.get("result", "")
        _emit({"node": spec.name, "text": f"{spec.name} finished: {result[:70]}"}, writer)
        return {
            "worker_results": [{"agent": spec.name, "result": result, "data": out.get("data", {})}],
            "steps": [{"node": spec.name, "text": result[:100]}],
        }

    return node
