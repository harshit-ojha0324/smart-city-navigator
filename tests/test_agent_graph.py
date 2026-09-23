"""End-to-end tests of the LangGraph agent (deterministic offline planner)."""
import asyncio
import os
import tempfile

from navigator.agent.graph import _config, agent_session, run_once


def _answer(q, **kw):
    return asyncio.run(run_once(q, **kw))["answer"]


def test_route_question():
    a = _answer("How do I get from Times Square to Coney Island?")
    assert "Take the" in a and "Coney Island" in a


def test_status_question_names_line():
    a = _answer("Is the L train running?")
    assert a.startswith("The L train")


def test_ambiguous_question_asks_to_clarify():
    a = _answer("How do I get to 125 St from Union Square?").lower()
    assert "which" in a or "125" in a


def test_out_of_scope_redirects():
    a = _answer("What's the weather in Paris?").lower()
    assert "subway" in a or "transit" in a


def test_bad_endpoints_do_not_fabricate_a_route():
    a = _answer("How do I get from Atlantis to Narnia?")
    assert "Take the" not in a
    assert "couldn't" in a.lower() or "no station" in a.lower()


def test_state_carries_intent_and_steps():
    st = asyncio.run(run_once("Is the subway running?"))
    assert st["intent"] in {"status", "info", "route", "other"}
    assert len(st["steps"]) >= 2  # understand + synthesize at minimum


def test_checkpoint_persists_to_disk_and_resumes():
    """Proves SqliteSaver fault-tolerance: a run written by one session is
    recoverable by a *fresh* session pointed at the same sqlite file."""
    path = tempfile.mktemp(suffix=".sqlite")

    async def run():
        async with agent_session(checkpoint_path=path) as g:
            await g.ainvoke(
                {"question": "Is the L train running?", "messages": [], "steps": []},
                config=_config("thread-persist"),
            )
        # Fresh session, same file — the checkpoint must be readable.
        async with agent_session(checkpoint_path=path) as g2:
            return await g2.aget_state(_config("thread-persist"))

    snap = asyncio.run(run())
    assert snap.values.get("answer", "").startswith("The L train")
    os.path.exists(path) and os.remove(path)
