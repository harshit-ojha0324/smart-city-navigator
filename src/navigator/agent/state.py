"""
Typed state for the transit-planning graph.

`messages` uses LangGraph's add_messages reducer so tool-calling turns accumulate
correctly; `steps` uses operator.add so every node can append a human-readable
reasoning line that the SSE gateway streams to the client.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Supervisor state: understand → supervise ⇄ worker agents → synthesize.

    `worker_results` accumulates (operator.add) one entry per specialist agent the
    supervisor delegates to, so a compound question can collect answers from
    several agents before synthesis.
    """

    question: str
    messages: Annotated[list, add_messages]
    intent: str
    route: str
    worker_results: Annotated[list[dict[str, Any]], operator.add]
    steps: Annotated[list[dict[str, Any]], operator.add]
    answer: str


class WorkerState(TypedDict):
    """Private state of one specialist worker agent's own plan → tools → finish loop."""

    task: str
    messages: Annotated[list, add_messages]
    result: str
    data: dict[str, Any]
