"""
Typed state for the transit-planning graph.

`messages` uses LangGraph's add_messages reducer so the conversation accumulates
across turns; `steps` and `worker_results` use the per_turn reducer so every node
can append within a turn, and each new question starts them fresh.
"""
from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


def per_turn(left: list | None, right: list | None) -> list:
    """Append within a turn; `None` resets.

    Plain operator.add would carry one question's worker results and steps into
    the next question on the same checkpointed thread_id (the supervisor would
    then see the Route Planner as already "done" and replay the old answer).
    Each new turn passes None to start these fields fresh, while `messages`
    keeps the conversation history.
    """
    if right is None:
        return []
    return (left or []) + right


class AgentState(TypedDict):
    """Supervisor state: understand → supervise ⇄ worker agents → synthesize.

    `worker_results` accumulates one entry per specialist agent the supervisor
    delegates to within a turn, so a compound question can collect answers from
    several agents before synthesis (reset between turns — see per_turn).
    """

    question: str
    messages: Annotated[list, add_messages]
    intent: str
    needs_status: bool | None   # LLM router: trip question that also asks about service
    router: str                 # "llm" or "planner" — which one classified the turn
    route: str
    worker_results: Annotated[list[dict[str, Any]], per_turn]
    steps: Annotated[list[dict[str, Any]], per_turn]
    answer: str


class WorkerState(TypedDict):
    """Private state of one specialist worker agent's own plan → tools → finish loop.

    `context_lines` is the supervisor's hand-off from an earlier agent (the lines
    of a route the Route Planner already found) so the next agent can act on it.
    """

    task: str
    messages: Annotated[list, add_messages]
    context_lines: list[str]
    result: str
    data: dict[str, Any]
