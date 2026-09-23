"""
The LLM path, exercised deterministically.

A real model can't run in CI, so these drive the same code with a scripted
chat model: the graph asks it to route the question and to pick tools, and the
tests assert on what the agent does with the replies — including the replies
that come back malformed, which is where an LLM-driven agent usually breaks.

`eval/run_eval.py --set all` runs the identical path against a real model
(Gemini, an OpenAI-compatible endpoint, or a local Ollama one); the README
records those scores.
"""
import asyncio

import pytest
from langchain_core.messages import AIMessage

from navigator.agent import llm as llm_module
from navigator.agent.graph import _config, _turn_input, build_graph
from navigator.agent.nodes import classify_with_llm
from navigator.agent.tools import inprocess_tools


class ScriptedModel:
    """Stands in for a chat model.

    `script` maps "router" / "worker" to a reply, or a list of replies consumed
    in order; once a worker's list runs out it answers with empty content, which
    ends that agent's tool loop. Replies are rebuilt per call with fresh ids —
    LangGraph dedupes messages by id, so handing back the same object twice
    silently drops the second turn. Every call is recorded for assertions.
    """

    def __init__(self, script):
        self.script = dict(script)
        self.calls: list[list] = []
        self.bound_tools = None

    def bind_tools(self, tools):
        self.bound_tools = tools
        return self

    @staticmethod
    def _fresh(message: AIMessage, n: int) -> AIMessage:
        calls = [{**c, "id": f"{c['id']}-{n}"} for c in (message.tool_calls or [])]
        return AIMessage(content=message.content, tool_calls=calls)

    async def ainvoke(self, messages, **_kwargs):
        self.calls.append(messages)
        system = next((m.content for m in messages if m.type == "system"), "")
        key = "router" if system.startswith("You route questions") else "worker"
        reply = self.script.get(key)
        if isinstance(reply, list):
            reply = reply.pop(0) if reply else AIMessage(content="")
        if isinstance(reply, Exception):
            raise reply
        if key == "worker" and not isinstance(self.script.get(key), list):
            # A single scripted worker reply is used once, then the loop ends.
            self.script[key] = []
        return self._fresh(reply, len(self.calls))


def _run(question, model):
    graph = build_graph(inprocess_tools(), llm=model).compile()
    return asyncio.run(graph.ainvoke(_turn_input(question), config=_config("llm-test")))


def _router(payload):
    return AIMessage(content=payload)


def _line_status_call():
    return AIMessage(content="", tool_calls=[
        {"name": "get_line_status", "args": {"line": "L"}, "id": "c1", "type": "tool_call"}])


# ── routing ────────────────────────────────────────────────────────────
def test_llm_routes_the_question():
    model = ScriptedModel({
        "router": _router('{"intent": "status", "needs_status": false}'),
        "worker": AIMessage(content="", tool_calls=[
            {"name": "get_line_status", "args": {"line": "L"}, "id": "c1", "type": "tool_call"}]),
    })
    state = _run("is the L ok", model)
    assert state["intent"] == "status" and state["router"] == "llm"
    assert [w["agent"] for w in state["worker_results"]] == ["service_advisor"]


def test_llm_router_can_chain_two_agents():
    """needs_status turns a trip question into a two-agent job."""
    model = ScriptedModel({
        "router": _router('{"intent": "route", "needs_status": true}'),
        "worker": AIMessage(content="", tool_calls=[
            {"name": "plan_trip", "args": {"origin": "Times Square", "destination": "Coney Island"},
             "id": "c1", "type": "tool_call"}]),
    })
    state = _run("times square to coney island, any issues", model)
    assert [w["agent"] for w in state["worker_results"]] == ["route_planner", "service_advisor"]


def test_llm_answer_is_used_when_the_model_writes_one():
    model = ScriptedModel({
        "router": _router('{"intent": "status", "needs_status": false}'),
        "worker": [AIMessage(content="", tool_calls=[
            {"name": "get_line_status", "args": {"line": "L"}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content="The L is running normally right now.")],
    })
    state = _run("is the L ok", model)
    assert state["answer"] == "The L is running normally right now."


def test_model_is_given_only_its_agents_tools():
    model = ScriptedModel({
        "router": _router('{"intent": "info", "needs_status": false}'),
        "worker": AIMessage(content="", tool_calls=[
            {"name": "resolve_station", "args": {"query": "Jay St"}, "id": "c1", "type": "tool_call"}]),
    })
    _run("which trains at Jay St", model)
    # Station Info owns lookup tools and must not be handed trip planning.
    names = {t.name for t in model.bound_tools}
    assert "resolve_station" in names and "plan_trip" not in names


# ── the model misbehaving ──────────────────────────────────────────────
@pytest.mark.parametrize("reply", [
    AIMessage(content="I think this is a routing question, friend."),  # prose, no JSON
    AIMessage(content='{"intent": "banana", "needs_status": false}'),  # invalid intent
    AIMessage(content='{"intent": '),                                   # truncated JSON
])
def test_unusable_router_replies_fall_back_to_the_planner(reply):
    model = ScriptedModel({"router": reply, "worker": _line_status_call()})
    state = _run("Is the L train running?", model)
    assert state["router"] == "planner"
    assert state["intent"] == "status"          # regex got it right anyway
    assert state["answer"].startswith("The L train")


def test_router_failure_does_not_break_the_turn():
    model = ScriptedModel({"router": RuntimeError("connection refused"),
                           "worker": _line_status_call()})
    state = _run("Is the L train running?", model)
    assert state["router"] == "planner" and state["answer"].startswith("The L train")


def test_classify_returns_none_rather_than_guessing():
    model = ScriptedModel({"router": AIMessage(content="no json here")})
    assert asyncio.run(classify_with_llm(model, "anything")) is None


def test_model_reply_in_content_parts_is_read():
    """Gemini can answer with a list of content blocks instead of a string."""
    model = ScriptedModel({
        "router": AIMessage(content=[{"type": "text",
                                      "text": '{"intent": "status", "needs_status": false}'}]),
        "worker": [AIMessage(content="", tool_calls=[
            {"name": "get_line_status", "args": {"line": "L"}, "id": "c1", "type": "tool_call"}]),
            AIMessage(content=[{"type": "text", "text": "The L is fine."}])],
    })
    state = _run("is the L ok", model)
    assert state["router"] == "llm" and state["answer"] == "The L is fine."


def test_endless_tool_calling_is_capped_not_looped():
    """A model that answers every turn with another tool call (seen with a local
    8B model) used to run until the graph's recursion limit and kill the turn."""
    model = ScriptedModel({
        "router": _router('{"intent": "status", "needs_status": false}'),
        # Never stops asking for tools.
        "worker": [_line_status_call() for _ in range(50)],
    })
    state = _run("is the L ok", model)
    assert state["answer"].startswith("The L train")   # composed from tool results
    worker_calls = [c for c in model.calls
                    if not next((m.content for m in c if m.type == "system"), "")
                    .startswith("You route questions")]
    assert len(worker_calls) <= 4


def test_a_failing_agent_does_not_take_down_the_turn():
    class Exploding(ScriptedModel):
        async def ainvoke(self, messages, **kwargs):
            system = next((m.content for m in messages if m.type == "system"), "")
            if not system.startswith("You route questions"):
                raise RuntimeError("model went away mid-answer")
            return await super().ainvoke(messages, **kwargs)

    model = Exploding({"router": _router('{"intent": "status", "needs_status": false}')})
    state = _run("is the L ok", model)
    assert "couldn't complete that lookup" in state["answer"]


# ── provider selection ─────────────────────────────────────────────────
def test_no_configuration_means_the_offline_planner(monkeypatch):
    for var in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NAVIGATOR_LLM_PROVIDER", "auto")
    assert llm_module.get_chat_model() is None
    assert "deterministic planner" in llm_module.describe_reasoner()


def test_a_key_selects_gemini(monkeypatch):
    monkeypatch.setenv("NAVIGATOR_LLM_PROVIDER", "auto")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    assert llm_module.active_provider() == "gemini"
    assert llm_module.describe_reasoner() == "gemini:gemini-2.5-flash"


def test_a_broken_provider_degrades_instead_of_raising(monkeypatch):
    monkeypatch.setenv("NAVIGATOR_LLM_PROVIDER", "nonsense-provider")
    assert llm_module.get_chat_model() is None
