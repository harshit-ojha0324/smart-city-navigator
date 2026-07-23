"""Tests pinning the supervisor + specialist-worker multi-agent architecture."""
import asyncio

from navigator.agent.graph import build_graph, run_once
from navigator.agent.tools import inprocess_tools
from navigator.agent.workers import WORKER_SPECS


def _agents(question: str):
    st = asyncio.run(run_once(question))
    return [w["agent"] for w in st["worker_results"]], st["answer"]


def test_three_specialist_agents_exist():
    names = {s.name for s in WORKER_SPECS}
    assert names == {"route_planner", "service_advisor", "station_info"}
    # Each worker owns a distinct, non-empty tool slice and its own persona.
    for spec in WORKER_SPECS:
        assert spec.tool_names and spec.persona


def test_graph_has_supervisor_and_worker_nodes():
    g = build_graph(inprocess_tools()).compile()
    nodes = set(g.get_graph().nodes.keys())
    assert {"understand", "supervise", "synthesize"} <= nodes
    assert {"route_planner", "service_advisor", "station_info"} <= nodes


def test_supervisor_routes_to_route_planner():
    agents, answer = _agents("How do I get from Times Square to Coney Island?")
    assert agents == ["route_planner"]
    assert "Take the" in answer


def test_supervisor_routes_to_service_advisor():
    agents, _ = _agents("Is the L train running?")
    assert agents == ["service_advisor"]


def test_supervisor_routes_to_station_info():
    agents, _ = _agents("Which lines stop at Atlantic Av?")
    assert agents == ["station_info"]


def test_compound_query_invokes_two_agents():
    agents, answer = _agents("How do I get from Penn Station to Fulton St and are there delays?")
    assert agents == ["route_planner", "service_advisor"]
    assert "Take the" in answer            # route planner contributed
    assert "everity" in answer or "Delays" in answer or "normally" in answer  # service advisor contributed


def test_out_of_scope_delegates_to_no_agent():
    agents, answer = _agents("What's the weather in Paris?")
    assert agents == []
    assert "subway" in answer.lower() or "transit" in answer.lower()
