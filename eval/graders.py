"""
Graders for the eval suite.

Route answers are graded against ground truth, not vocabulary: the grader
resolves the endpoints and plans the trip itself with the routing engine, then
checks that the answer rides the same lines, ends at the same station, quotes a
time close to the computed one, and — the part keyword matching can't do —
names no line the real itinerary doesn't use. Status answers are checked
against what the feed actually reports for that line right now.

The remaining kinds (clarification, graceful failure, out-of-scope) are
behavioral: what matters is that the agent asked or declined instead of
inventing an itinerary, so those assert on shape plus the absence of a route.

Deterministic and offline, so pass/fail is reproducible in CI. The same logic
is exposed as a LangSmith evaluator for the tracing path.
"""
from __future__ import annotations

import re

from navigator.core import geocode, mta_feed, routing
from navigator.core.graph_data import SUBWAY_LINES, feed_line, line_label

_CLARIFY_MARKERS = ["which", "could be", "did you mean", "more detail", "several stations"]
# A refusal counts however it is worded — what matters is that no route was
# invented and the rider is told the thing wasn't found. Listing only the
# deterministic composer's phrasing would mark a model's correct refusal wrong.
_FAIL_MARKERS = ["couldn't", "could not", "no route", "unknown", "no station", "don't",
                 "unable", "not found", "too far", "doesn't", "outside", "not a valid",
                 "isn't a valid", "is not a valid", "invalid", "does not serve",
                 "doesn't serve", "not served", "no such", "not recognized",
                 "not a station", "cannot", "can't"]
_SCOPE_MARKERS = ["subway", "transit", "trip planner", "station", "route"]
# An itinerary names lines to ride. Phrasings vary — "take the L", "transfer to
# the 7 express", "via the 5 line" — so the grader extracts the *sequence of
# lines* rather than insisting on one sentence shape. Grading the planner's own
# wording back to itself would flatter it and punish any model that writes
# differently while being right.
_LINE_MENTION = re.compile(
    r"\b(?:take|takes|transfer to|transferring to|then|via|board|catch|ride|switch to)\s+"
    r"(?:the\s+)?([A-Za-z0-9]{1,2}(?:\s+express)?)\b(?:\s+(?:line|train))?", re.I)
_MINUTES = re.compile(r"(\d+(?:\.\d+)?)\s*(?:min\b|mins\b|minutes\b)", re.I)
_VALID_LABELS = {line_label(line).lower() for line in SUBWAY_LINES}


def _contains_any(text: str, needles) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def grade(case: dict, state: dict) -> tuple[bool, str]:
    """Return (passed, reason) for one evaluated case."""
    answer = (state.get("answer") or "").strip()
    expect = case.get("expect", {})
    if not answer:
        return False, "empty answer"

    ok, reason = _grade_kind(expect.get("kind"), expect, answer)
    if ok and expect.get("mentions_any") and not _contains_any(answer, expect["mentions_any"]):
        return False, f"missing any of {expect['mentions_any']}"
    # Intent is a soft signal — recorded, but `kind` decides.
    if expect.get("intent") and state.get("intent") != expect["intent"]:
        reason += f" (intent {state.get('intent')}!={expect['intent']})"
    return ok, reason


def _grade_kind(kind, expect, answer) -> tuple[bool, str]:
    if kind == "route":
        return _grade_route(expect, answer)
    if kind == "line_status":
        return _grade_line_status(expect, answer)

    low = answer.lower()

    if kind == "service_status":
        if not _contains_any(low, ["good service", "delays", "planned work", "service change",
                                   "suspended", "running normally", "issues", "schedule"]):
            return False, "no service summary language"
        return True, "service status ok"

    if kind == "station_info":
        return _grade_station_info(expect, answer)

    if kind == "geocode":
        return _grade_geocode(expect, answer)

    if kind == "clarify":
        if _lines_named(answer):
            return False, "planned a trip instead of asking which station"
        if not _contains_any(low, _CLARIFY_MARKERS):
            return False, "did not ask for clarification"
        term = expect.get("term")
        if term:
            resolved = geocode.resolve_station(term)
            if not resolved["ambiguous"]:
                return False, f"dataset error: {term!r} is not ambiguous"
            named = sum(1 for sid in resolved["candidates"]
                        if geocode.STATION_BY_ID[sid]["name"].lower() in low)
            if named < 2:
                return False, "did not name the stations to choose between"
        return True, "asked to disambiguate"

    if kind == "needs_origin":
        if _lines_named(answer):
            return False, "invented a trip with no origin given"
        if not _contains_any(low, ["starting from", "where are you", "origin",
                                   "starting location", "starting point", "start from",
                                   "where you", "your starting"]):
            return False, "did not ask where the rider is starting"
        return True, "asked for the origin"

    if kind == "graceful_fail":
        if _lines_named(answer):
            return False, "fabricated a route on bad input"
        if not _contains_any(low, _FAIL_MARKERS):
            return False, "no graceful failure language"
        return True, "failed gracefully"

    if kind == "degenerate":
        return True, "handled degenerate trip"

    if kind == "out_of_scope":
        if _lines_named(answer):
            return False, "answered an out-of-scope question with a trip"
        if not _contains_any(low, _SCOPE_MARKERS):
            return False, "did not redirect to transit"
        return True, "redirected out-of-scope query"

    if kind == "elevator":
        if not _contains_any(low, ["elevator", "escalator"]):
            return False, "did not address elevator/escalator status"
        return True, "answered on accessibility equipment"

    return True, "no kind assertion"


# ── ground-truth graders ───────────────────────────────────────────────
def _lines_named(answer: str) -> list[str]:
    """The lines an answer tells the rider to ride, in order, however phrased."""
    found: list[str] = []
    for raw in _LINE_MENTION.findall(answer):
        label = re.sub(r"\s+", " ", raw.strip()).lower()
        if label in _VALID_LABELS and (not found or found[-1] != label):
            found.append(label)
    return found


def _grade_route(expect, answer) -> tuple[bool, str]:
    """Compare the answer with the itinerary the engine computes independently."""
    try:
        origin = geocode.resolve_endpoint(expect["origin"])
        destination = geocode.resolve_endpoint(expect["destination"])
        truth = routing.plan_route(origin["id"], destination["id"])
    except (geocode.GeocodeError, routing.RouteError) as exc:
        return False, f"dataset error: cannot plan the reference trip ({exc})"

    expected = [line_label(leg["line"]).lower() for leg in truth["legs"] if not leg["walk"]]
    named = _lines_named(answer)
    if not named:
        return False, "no itinerary in the answer"
    if named != expected:
        return False, f"lines {named} != planned {expected}"
    if destination["name"].lower() not in answer.lower():
        return False, f"answer never reaches {destination['name']}"

    quoted = [float(m) for m in _MINUTES.findall(answer)]
    if not quoted:
        return False, "no travel time quoted"
    # Answers may break time down per leg; one of the figures must be the total.
    if not any(abs(minutes - truth["total_minutes"]) <= 2 for minutes in quoted):
        return False, f"quoted {quoted} min vs planned {truth['total_minutes']}"
    return True, f"matches planned trip ({truth['total_minutes']} min, {len(expected)} leg(s))"


def _grade_line_status(expect, answer) -> tuple[bool, str]:
    """The answer must report what the feed actually says for that line."""
    line = expect["line"]
    truth = mta_feed.get_line_status(line)
    low = answer.lower()
    if truth.get("error"):  # e.g. "QZ" — the agent must not invent a status
        if _contains_any(low, _FAIL_MARKERS) or "unknown" in low:
            return True, "reported the line as unknown"
        return False, "invented a status for a line that does not exist"
    if not re.search(rf"\b{re.escape(line)}\b", answer):
        return False, f"line {line} not named"
    if truth["status"].lower() not in low:
        return False, f"said something other than the feed's {truth['status']!r}"
    return True, f"matches feed ({truth['status']})"


def _grade_station_info(expect, answer) -> tuple[bool, str]:
    """Every line the station complex serves must be named, and no others."""
    station = expect.get("station")
    if not station:
        return True, "station info ok"
    resolved = geocode.resolve_station(station)
    if resolved["station"]["name"].lower() not in answer.lower():
        return False, f"did not name {resolved['station']['name']}"
    truth = {feed_line(line) for line in geocode.complex_lines(resolved["station"]["id"])}
    quoted = set(re.findall(r"\b([1-7]|[A-Z])\b(?=[,.\s])", answer)) & set("1234567ABCDEFGJLMNQRSWZ")
    missing = truth - quoted
    if missing and not resolved["ambiguous"]:
        return False, f"missed lines {sorted(missing)}"
    return True, "named the lines serving the station"


def _grade_geocode(expect, answer) -> tuple[bool, str]:
    """The station named must be the one nearest the place."""
    place = expect["place"]
    truth = geocode.geocode_place(place)["nearest_station"]["name"]
    if truth.lower() not in answer.lower():
        return False, f"nearest station is {truth}, answer said otherwise"
    return True, f"nearest station {truth}"


# ── LangSmith evaluator adapter ────────────────────────────────────────
def langsmith_correctness(run, example) -> dict:
    """LangSmith evaluator: score 1/0 using the same graders."""
    outputs = run.outputs or {}
    state = outputs.get("state") or {"answer": outputs.get("answer", ""),
                                     "intent": outputs.get("intent", "")}
    case = (example.metadata or {}).get("case") or {"expect": (example.outputs or {})}
    passed, reason = grade(case, state)
    return {"key": "correctness", "score": 1 if passed else 0, "comment": reason}
