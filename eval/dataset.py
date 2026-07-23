"""
20-prompt evaluation set for the Smart City Navigator agent.

Three categories, matching the resume:
  * happy_path    (10) — routes / status / info questions that should succeed
  * ambiguous      (5) — endpoints that match several stations; the agent must
                         ask which one rather than guess
  * tool_failure   (5) — unknown places/lines, degenerate trips, feed outages,
                         and out-of-scope questions; the agent must degrade
                         gracefully and never fabricate a route

The whole suite runs with the live GTFS-RT feed stubbed (NAVIGATOR_SIMULATE_FEED),
so every status answer also exercises the fault-tolerant simulation fallback.

Each case carries a light `expect` spec consumed by eval/graders.py.
"""
from __future__ import annotations

EVAL_CASES: list[dict] = [
    # ── happy_path ─────────────────────────────────────────────────────
    {"id": "hp01", "category": "happy_path",
     "question": "How do I get from Times Square to Coney Island?",
     "expect": {"intent": "route", "kind": "route", "mentions_any": ["Coney Island", "stops", "min"]}},
    {"id": "hp02", "category": "happy_path",
     "question": "from Penn Station to Fulton St",
     "expect": {"intent": "route", "kind": "route", "mentions_any": ["Fulton", "stops"]}},
    {"id": "hp03", "category": "happy_path",
     "question": "What's the fastest way from Grand Central to Union Square?",
     "expect": {"intent": "route", "kind": "route", "mentions_any": ["Union Sq", "stops", "min"]}},
    {"id": "hp04", "category": "happy_path",
     "question": "Get me from World Trade Center to Flushing Main St",
     "expect": {"intent": "route", "kind": "route", "mentions_any": ["transfer", "stops", "min"]}},
    {"id": "hp05", "category": "happy_path",
     "question": "from Bedford Av to Herald Sq",
     "expect": {"intent": "route", "kind": "route", "mentions_any": ["Herald", "stops"]}},
    {"id": "hp06", "category": "happy_path",
     "question": "Is the L train running?",
     "expect": {"intent": "status", "kind": "line_status", "line": "L"}},
    {"id": "hp07", "category": "happy_path",
     "question": "What's the subway status right now?",
     "expect": {"intent": "status", "kind": "service_status"}},
    {"id": "hp08", "category": "happy_path",
     "question": "Are there delays on the 7?",
     "expect": {"intent": "status", "kind": "line_status", "line": "7"}},
    {"id": "hp09", "category": "happy_path",
     "question": "Which lines stop at Atlantic Av?",
     "expect": {"intent": "info", "kind": "station_info", "mentions_any": ["Atlantic"]}},
    {"id": "hp10", "category": "happy_path",
     "question": "What's the nearest station to the Empire State Building?",
     "expect": {"intent": "info", "kind": "geocode", "mentions_any": ["Herald", "km"]}},

    # ── ambiguous ──────────────────────────────────────────────────────
    {"id": "am01", "category": "ambiguous",
     "question": "How do I get to 125 St from Union Square?",
     "expect": {"intent": "route", "kind": "clarify"}},
    {"id": "am02", "category": "ambiguous",
     "question": "from 125 St to Times Square",
     "expect": {"intent": "route", "kind": "clarify"}},
    {"id": "am03", "category": "ambiguous",
     "question": "How do I get from 23 St to Fulton St?",
     "expect": {"intent": "route", "kind": "clarify"}},
    {"id": "am04", "category": "ambiguous",
     "question": "Plan a trip from 28 St to Union Square",
     "expect": {"intent": "route", "kind": "clarify"}},
    {"id": "am05", "category": "ambiguous",
     "question": "Which lines stop at 125 St?",
     "expect": {"intent": "info", "kind": "station_info", "mentions_any": ["ambiguous", "125"]}},

    # ── tool_failure / robustness ──────────────────────────────────────
    {"id": "tf01", "category": "tool_failure",
     "question": "How do I get from Atlantis to Narnia?",
     "expect": {"intent": "route", "kind": "graceful_fail", "not_mentions": ["Take the"]}},
    {"id": "tf02", "category": "tool_failure",
     "question": "How do I get from Times Square to Times Square?",
     "expect": {"intent": "route", "kind": "degenerate", "mentions_any": ["already"]}},
    {"id": "tf03", "category": "tool_failure",
     "question": "Is the QZ train running?",
     "expect": {"intent": "status", "kind": "graceful_fail"}},
    {"id": "tf04", "category": "tool_failure",
     "question": "What's the status of the A train?",
     "expect": {"intent": "status", "kind": "degraded_ok", "line": "A"}},
    {"id": "tf05", "category": "tool_failure",
     "question": "What's the weather in Paris?",
     "expect": {"intent": "other", "kind": "out_of_scope"}},
]


CATEGORIES = ("happy_path", "ambiguous", "tool_failure")


def cases_for(category: str | None = None) -> list[dict]:
    if category in (None, "all"):
        return list(EVAL_CASES)
    return [c for c in EVAL_CASES if c["category"] == category]
