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

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import StreamWriter

from .llm import DeterministicPlanner
from .nodes import _emit, compose_answer, last_ai, tool_results
from .state import WorkerState


@dataclass
class WorkerSpec:
    name: str
    persona: str
    tool_names: tuple[str, ...]
    intent: str  # intent label the deterministic policy plans under


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
        if llm_with_tools is not None:
            ai = await llm_with_tools.ainvoke([SystemMessage(content=spec.persona)] + messages)
            return {"messages": [ai]}
        if not results_in:
            calls = planner.initial_tool_calls(state["task"], spec.intent)
            return {"messages": [AIMessage(content="", tool_calls=calls)]}
        return {"messages": [AIMessage(content="")]}

    async def finish(state) -> dict:
        messages = state["messages"]
        last = last_ai(messages)
        if llm is not None and last is not None and last.content and not getattr(last, "tool_calls", None):
            result, data = last.content, {}
        else:
            results = tool_results(messages)
            result = compose_answer(spec.intent, results, state["task"])
            data = results[-1][1] if results else {}
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


def make_worker_node(spec: WorkerSpec, subgraph):
    """A supervisor node that delegates the task to one worker agent's subgraph."""
    async def node(state, writer: StreamWriter = None) -> dict:
        _emit({"node": spec.name, "text": f"Supervisor → {spec.name}: delegating the task."}, writer)
        out = await subgraph.ainvoke({
            "task": state["question"],
            "messages": [HumanMessage(content=state["question"])],
            "result": "", "data": {},
        })
        result = out.get("result", "")
        _emit({"node": spec.name, "text": f"{spec.name} finished: {result[:70]}"}, writer)
        return {
            "worker_results": [{"agent": spec.name, "result": result, "data": out.get("data", {})}],
            "steps": [{"node": spec.name, "text": result[:100]}],
        }

    return node
