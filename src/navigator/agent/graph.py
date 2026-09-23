"""
Supervisor multi-agent graph.

    understand ──▶ supervise ──▶ route_planner  ─┐
                     ▲   │  ├──▶ service_advisor ─┤
                     │   │  └──▶ station_info   ──┤
                     │   └──────────────────────────▶ synthesize ──▶ END
                     └───────────(loop back)────────┘

A supervisor delegates each question to one or more independent specialist worker
agents (each its own subgraph with its own persona/tools/loop — see workers.py),
looping back so a compound question can visit several agents, then a synthesis
step merges their results. Compiled with an AsyncSqliteSaver checkpointer so a run
is resumable by thread_id after a crash.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from langgraph.graph import END, StateGraph

from .llm import _STATUS_WORDS, DeterministicPlanner, get_chat_model
from .nodes import compose_answer, make_understand_node
from .state import AgentState
from .tools import build_tools
from .workers import WORKER_SPECS, build_worker, make_worker_node

# intent → ordered list of specialist agents the supervisor delegates to
_PLAN = {
    "route": ["route_planner"],
    "status": ["service_advisor"],
    "info": ["station_info"],
    "other": [],
}


def _needed_workers(state) -> list[str]:
    plan = list(_PLAN.get(state.get("intent", "other"), []))
    # Compound query: a trip question that also asks about delays/service gets the
    # Service Advisor after the Route Planner — genuine multi-agent collaboration.
    # The LLM router says so directly; the regex planner infers it from wording.
    flagged = state.get("needs_status")
    if flagged is None:
        flagged = bool(_STATUS_WORDS.search(state.get("question", "")))
    if state.get("intent") == "route" and flagged:
        plan = ["route_planner", "service_advisor"]
    return plan


def make_supervisor_node():
    async def supervise(state) -> dict:
        done = {w["agent"] for w in state.get("worker_results", [])}
        nxt = next((w for w in _needed_workers(state) if w not in done), "synthesize")
        return {"route": nxt,
                "steps": [{"node": "supervisor", "text": f"Supervisor routing to: {nxt}"}]}

    return supervise


def make_synthesize_node():
    async def synthesize(state) -> dict:
        results = state.get("worker_results", [])
        if not results:
            answer = compose_answer(state.get("intent", "other"), [], state["question"])
        elif len(results) == 1:
            answer = results[0]["result"]
        else:
            answer = "\n\n".join(r["result"] for r in results if r.get("result"))
        return {"answer": answer,
                "steps": [{"node": "synthesize",
                           "text": "Merged agent results into the final answer."}]}

    return synthesize


def build_graph(tools, llm=None, planner: DeterministicPlanner | None = None) -> StateGraph:
    """Uncompiled supervisor StateGraph wiring the understand → supervise ⇄ workers → synth topology."""
    planner = planner or DeterministicPlanner()
    builder = StateGraph(AgentState)
    builder.add_node("understand", make_understand_node(planner, llm))
    builder.add_node("supervise", make_supervisor_node())
    builder.add_node("synthesize", make_synthesize_node())

    worker_targets = {}
    for spec in WORKER_SPECS:
        subgraph = build_worker(spec, tools, llm, planner)
        builder.add_node(spec.name, make_worker_node(spec, subgraph))
        builder.add_edge(spec.name, "supervise")  # report back to the supervisor
        worker_targets[spec.name] = spec.name

    builder.set_entry_point("understand")
    builder.add_edge("understand", "supervise")
    builder.add_conditional_edges(
        "supervise", lambda s: s["route"],
        {**worker_targets, "synthesize": "synthesize"},
    )
    builder.add_edge("synthesize", END)
    return builder


async def build_compiled(transport: str = "inprocess", host: str | None = None, checkpointer=None):
    tools = await build_tools(transport, host)
    llm = get_chat_model()
    return build_graph(tools, llm).compile(checkpointer=checkpointer)


@asynccontextmanager
async def agent_session(transport: str = "inprocess", host: str | None = None,
                        checkpoint_path: str = ":memory:"):
    """Async context manager yielding a compiled graph with SqliteSaver checkpointing."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        graph = await build_compiled(transport, host, checkpointer=saver)
        yield graph


def _turn_input(question: str) -> dict:
    """Input for one question. None resets the per-turn fields (see state.per_turn)
    so a reused, checkpointed thread_id can't replay a previous turn's results."""
    return {"question": question, "worker_results": None, "steps": None,
            "intent": "", "route": "", "answer": "", "needs_status": None, "router": ""}


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 30}


async def run_once(question: str, *, thread_id: str = "default",
                   transport: str = "inprocess", host: str | None = None,
                   checkpoint_path: str = ":memory:") -> dict:
    """Answer one question end-to-end; returns the final state dict."""
    async with agent_session(transport, host, checkpoint_path) as graph:
        return await graph.ainvoke(
            _turn_input(question),
            config=_config(thread_id))


async def stream_once(question: str, *, thread_id: str = "default",
                      transport: str = "inprocess", host: str | None = None,
                      checkpoint_path: str = ":memory:"):
    """Yield reasoning events (dicts) as the agents work, then a final 'answer' event.

    The events are the `steps` each node appends to state, streamed in
    `updates` mode as that node finishes. An earlier version pushed them
    through LangGraph's custom stream writer, which depends on contextvars
    reaching the node's task — that silently stopped working for async nodes,
    and a stream that fails by going quiet is worse than one that never
    existed. Reading them off the state updates has no such dependency.
    """
    async with agent_session(transport, host, checkpoint_path) as graph:
        final_state = None
        async for mode, chunk in graph.astream(
            _turn_input(question),
            config=_config(thread_id), stream_mode=["updates", "values"],
        ):
            if mode == "updates":
                for update in (chunk or {}).values():
                    for step in (update or {}).get("steps") or []:
                        yield step
            elif mode == "values":
                final_state = chunk
        yield {"node": "done", "answer": (final_state or {}).get("answer", "")}
