"""
Heuristic graders for the eval suite.

Each grader inspects the agent's final answer (and intent) against the case's
`expect` spec. Deterministic and offline so pass/fail is reproducible in CI.
The same logic is exposed as a LangSmith evaluator for the tracing path.
"""
from __future__ import annotations

_CLARIFY_MARKERS = ["which", "ambiguous", "could be", "did you mean", "more detail", "which one"]
_FAIL_MARKERS = ["couldn't", "could not", "no route", "unknown", "no station", "don't", "unable", "not found"]
_SCOPE_MARKERS = ["subway", "transit", "trip planner", "station", "route"]
_STATUS_WORDS = ["good service", "delays", "severity", "suspended", "planned work",
                 "service change", "running", "schedule"]


def _contains_any(text: str, needles) -> bool:
    t = text.lower()
    return any(n.lower() in t for n in needles)


def grade(case: dict, state: dict) -> tuple[bool, str]:
    """Return (passed, reason) for one evaluated case."""
    answer = (state.get("answer") or "").strip()
    intent = state.get("intent", "")
    expect = case.get("expect", {})
    kind = expect.get("kind")

    if not answer:
        return False, "empty answer"

    # Intent is a soft signal — record a mismatch but let `kind` decide.
    intent_note = ""
    if expect.get("intent") and intent != expect["intent"]:
        intent_note = f" (intent {intent}!={expect['intent']})"

    ok, reason = _grade_kind(kind, expect, answer, intent)
    return ok, (reason + intent_note)


def _grade_kind(kind, expect, answer, intent) -> tuple[bool, str]:
    low = answer.lower()

    if kind == "route":
        if not _contains_any(low, ["take the", " min", "stops", "transfer"]):
            return False, "route answer lacks itinerary language"
        return _check_mentions(expect, low, "route ok")

    if kind == "line_status":
        line = expect.get("line", "")
        if line and line.lower() not in low:
            return False, f"line {line} not named"
        if not _contains_any(low, _STATUS_WORDS):
            return False, "no status language"
        return True, "line status ok"

    if kind == "service_status":
        if not _contains_any(low, _STATUS_WORDS + ["normally", "issues"]):
            return False, "no service summary language"
        return True, "service status ok"

    if kind == "station_info":
        return _check_mentions(expect, low, "station info ok")

    if kind == "geocode":
        return _check_mentions(expect, low, "geocode ok")

    if kind == "clarify":
        if not _contains_any(low, _CLARIFY_MARKERS):
            return False, "did not ask for clarification"
        return True, "asked to disambiguate"

    if kind == "graceful_fail":
        if "take the" in low:
            return False, "fabricated a route on bad input"
        if not _contains_any(low, _FAIL_MARKERS):
            return False, "no graceful failure language"
        return _check_not_mentions(expect, low, "failed gracefully")

    if kind == "degenerate":
        return _check_mentions(expect, low, "handled degenerate trip")

    if kind == "degraded_ok":
        line = expect.get("line", "")
        if line and line.lower() not in low:
            return False, f"line {line} not named"
        if not _contains_any(low, _STATUS_WORDS):
            return False, "no status despite fallback"
        return True, "answered from degraded feed"

    if kind == "out_of_scope":
        if not _contains_any(low, _SCOPE_MARKERS):
            return False, "did not redirect to transit"
        return True, "redirected out-of-scope query"

    return True, "no kind assertion"


def _check_mentions(expect, low, ok_msg) -> tuple[bool, str]:
    needles = expect.get("mentions_any")
    if needles and not _contains_any(low, needles):
        return False, f"missing any of {needles}"
    return True, ok_msg


def _check_not_mentions(expect, low, ok_msg) -> tuple[bool, str]:
    banned = expect.get("not_mentions")
    if banned and _contains_any(low, banned):
        return False, f"contains banned {banned}"
    return True, ok_msg


# ── LangSmith evaluator adapter ────────────────────────────────────────
def langsmith_correctness(run, example) -> dict:
    """LangSmith evaluator: score 1/0 using the same heuristic graders."""
    outputs = run.outputs or {}
    state = outputs.get("state") or {"answer": outputs.get("answer", ""),
                                     "intent": outputs.get("intent", "")}
    case = (example.metadata or {}).get("case") or {"expect": (example.outputs or {})}
    passed, reason = grade(case, state)
    return {"key": "correctness", "score": 1 if passed else 0, "comment": reason}
