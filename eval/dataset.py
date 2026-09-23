"""
Evaluation prompts for the Smart City Navigator agent.

Four sets, kept apart on purpose:

  * CORE (20) — the suite the agent was developed against. Three categories:
      happy_path   (10) routes / status / info questions that should succeed
      ambiguous     (5) names that match several stations; the agent must ask
      tool_failure  (5) unknown places and lines, degenerate trips, a stubbed
                        feed, out-of-scope questions — degrade, never fabricate

  * HELD-OUT (20), FRESH (15), WILD (15) — three generations of the same idea.
    Each was written against the finished parser, in the words people actually
    type (typos, no punctuation, slang: "trains messed up on the 6?"), scored
    exactly once, and then *spent*: the gaps it found were fixed, so the parser
    has now seen it and its score no longer measures generalization. The next
    one gets written instead.

    First scores, in order: 15/20 → 11/15 → 13/15. Each generation finds less,
    and that trajectory — not the 100% they all show today — is the result.

    If you change the planner, write a new set. Re-running a spent one tells
    you only that you didn't regress.

Every set runs with the live GTFS-RT feed stubbed (NAVIGATOR_SIMULATE_FEED), so
status answers also exercise the fault-tolerant simulation fallback.

Each case carries an `expect` spec consumed by eval/graders.py. Route cases name
their endpoints rather than expected text: the grader resolves them and plans
the trip itself, then checks the answer against that ground truth.
"""
from __future__ import annotations

# Any of these count as "told the rider this name is shared" — the deterministic
# composer and a model will word it differently and both are correct.
AMBIGUITY_WORDS = ["several stations share", "several stations", "ambiguous",
                   "multiple stations", "more than one station", "other stations",
                   "nearby station", "also matches", "which one"]

CORE_CASES: list[dict] = [
    # ── happy_path ─────────────────────────────────────────────────────
    {"id": "hp01", "category": "happy_path",
     "question": "How do I get from Times Square to Coney Island?",
     "expect": {"intent": "route", "kind": "route",
                "origin": "Times Square", "destination": "Coney Island"}},
    {"id": "hp02", "category": "happy_path",
     "question": "from Penn Station to Union Square",
     "expect": {"intent": "route", "kind": "route",
                "origin": "Penn Station", "destination": "Union Square"}},
    {"id": "hp03", "category": "happy_path",
     "question": "What's the fastest way from Grand Central to Union Square?",
     "expect": {"intent": "route", "kind": "route",
                "origin": "Grand Central", "destination": "Union Square"}},
    {"id": "hp04", "category": "happy_path",
     "question": "Get me from World Trade Center to Flushing Main St",
     "expect": {"intent": "route", "kind": "route",
                "origin": "World Trade Center", "destination": "Flushing Main St"}},
    {"id": "hp05", "category": "happy_path",
     "question": "from Bedford Av to Herald Sq",
     "expect": {"intent": "route", "kind": "route",
                "origin": "Bedford Av", "destination": "Herald Sq"}},
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
     "question": "Which lines stop at Atlantic Av-Barclays Ctr?",
     "expect": {"intent": "info", "kind": "station_info", "station": "Atlantic Av-Barclays Ctr"}},
    {"id": "hp10", "category": "happy_path",
     "question": "What's the nearest station to the Empire State Building?",
     "expect": {"intent": "info", "kind": "geocode", "place": "Empire State Building"}},

    # ── ambiguous ──────────────────────────────────────────────────────
    {"id": "am01", "category": "ambiguous",
     "question": "How do I get to 125 St from Union Square?",
     "expect": {"intent": "route", "kind": "clarify", "term": "125 St"}},
    {"id": "am02", "category": "ambiguous",
     "question": "from 125 St to Times Square",
     "expect": {"intent": "route", "kind": "clarify", "term": "125 St"}},
    {"id": "am03", "category": "ambiguous",
     "question": "How do I get from 86 St to Fulton St?",
     "expect": {"intent": "route", "kind": "clarify", "term": "86 St"}},
    {"id": "am04", "category": "ambiguous",
     "question": "Plan a trip from 28 St to Union Square",
     "expect": {"intent": "route", "kind": "clarify", "term": "28 St"}},
    {"id": "am05", "category": "ambiguous",
     "question": "Which lines stop at 125 St?",
     "expect": {"intent": "info", "kind": "station_info", "station": "125 St",
                "mentions_any": AMBIGUITY_WORDS}},

    # ── tool_failure / robustness ──────────────────────────────────────
    {"id": "tf01", "category": "tool_failure",
     "question": "How do I get from Atlantis to Narnia?",
     "expect": {"intent": "route", "kind": "graceful_fail"}},
    {"id": "tf02", "category": "tool_failure",
     "question": "How do I get from Times Square to Times Square?",
     "expect": {"intent": "route", "kind": "degenerate", "mentions_any": ["already"]}},
    {"id": "tf03", "category": "tool_failure",
     "question": "Is the QZ train running?",
     "expect": {"intent": "status", "kind": "graceful_fail"}},
    {"id": "tf04", "category": "tool_failure",
     "question": "What's the status of the A train?",
     "expect": {"intent": "status", "kind": "line_status", "line": "A"}},
    {"id": "tf05", "category": "tool_failure",
     "question": "What's the weather in Paris?",
     "expect": {"intent": "other", "kind": "out_of_scope"}},
]


HELDOUT_CASES: list[dict] = [
    # Routes, asked the way people type them.
    {"id": "ho01", "category": "heldout_route",
     "question": "whats the quickest way to get to Coney Island from Times Square",
     "expect": {"kind": "route", "origin": "Times Square", "destination": "Coney Island"}},
    {"id": "ho02", "category": "heldout_route",
     "question": "i need to be at Grand Central, coming from Bedford Av",
     "expect": {"kind": "route", "origin": "Bedford Av", "destination": "Grand Central"}},
    {"id": "ho03", "category": "heldout_route",
     "question": "How many stops is it from Union Square to Brooklyn Bridge?",
     "expect": {"kind": "route", "origin": "Union Square", "destination": "Brooklyn Bridge"}},
    {"id": "ho04", "category": "heldout_route",
     "question": "Take me from Columbus Circle to Wall St please",
     "expect": {"kind": "route", "origin": "Columbus Circle", "destination": "Wall St"}},
    {"id": "ho05", "category": "heldout_route",
     "question": "Plan me a trip: Astoria Blvd to Herald Square",
     "expect": {"kind": "route", "origin": "Astoria Blvd", "destination": "Herald Square"}},
    {"id": "ho06", "category": "heldout_route",
     "question": "subway from Prospect Park to Rockefeller Center?",
     "expect": {"kind": "route", "origin": "Prospect Park", "destination": "Rockefeller Center"}},

    # Service status, in everyday words.
    {"id": "ho07", "category": "heldout_status",
     "question": "Is the 4 running okay this morning?",
     "expect": {"kind": "line_status", "line": "4"}},
    {"id": "ho08", "category": "heldout_status",
     "question": "any problems on the J?",
     "expect": {"kind": "line_status", "line": "J"}},
    {"id": "ho09", "category": "heldout_status",
     "question": "What's up with the G train?",
     "expect": {"kind": "line_status", "line": "G"}},
    {"id": "ho10", "category": "heldout_status",
     "question": "trains messed up on the 6?",
     "expect": {"kind": "line_status", "line": "6"}},
    {"id": "ho11", "category": "heldout_status",
     "question": "how's the subway doing",
     "expect": {"kind": "service_status"}},

    # Station / place lookups.
    {"id": "ho12", "category": "heldout_info",
     "question": "Which trains can I catch at Jay St?",
     "expect": {"kind": "station_info", "station": "Jay St"}},
    {"id": "ho13", "category": "heldout_info",
     "question": "Closest stop to Yankee Stadium?",
     "expect": {"kind": "geocode", "place": "Yankee Stadium"}},
    {"id": "ho14", "category": "heldout_info",
     "question": "what lines are at 86 St",
     "expect": {"kind": "station_info", "station": "86 St",
                "mentions_any": AMBIGUITY_WORDS}},

    # Robustness: things that must not produce a confident itinerary.
    {"id": "ho15", "category": "heldout_failure",
     "question": "Get from 96 St to Wall St",
     "expect": {"kind": "clarify", "term": "96 St"}},
    {"id": "ho16", "category": "heldout_failure",
     "question": "How do I get to Bedford Av?",
     "expect": {"kind": "needs_origin"}},
    {"id": "ho17", "category": "heldout_failure",
     "question": "Can I take the subway from Times Square to Hoboken?",
     "expect": {"kind": "graceful_fail"}},
    {"id": "ho18", "category": "heldout_failure",
     "question": "Is the 11 train delayed?",
     "expect": {"kind": "graceful_fail"}},
    {"id": "ho19", "category": "heldout_failure",
     "question": "Who won the Yankees game last night?",
     "expect": {"kind": "out_of_scope"}},
    {"id": "ho20", "category": "heldout_failure",
     "question": "elevator out at 14 St-Union Sq?",
     "expect": {"kind": "elevator"}},
]


# ── fresh (v2 held-out) ────────────────────────────────────────────────
# The held-out set above did its job — it scored 15/20 and exposed five real
# gaps, which were then fixed. That makes it tuned-on, so these 15 were written
# afterwards, against the finished planner, and scored once. When you next
# change the planner, this set is spent too: write another.
FRESH_CASES: list[dict] = [
    {"id": "fr01", "category": "fresh_route",
     "question": "wat time does it take from Astoria Blvd to Times Sq",
     "expect": {"kind": "route", "origin": "Astoria Blvd", "destination": "Times Sq"}},
    {"id": "fr02", "category": "fresh_route",
     "question": "im at Grand Central and need to reach Barclays Center",
     "expect": {"kind": "route", "origin": "Grand Central", "destination": "Barclays Center"}},
    {"id": "fr03", "category": "fresh_route",
     "question": "how long to get from Flushing Main St to Jamaica Center",
     "expect": {"kind": "route", "origin": "Flushing Main St", "destination": "Jamaica Center"}},
    {"id": "fr04", "category": "fresh_route",
     "question": "from Bushwick to Wall St",
     "expect": {"kind": "route", "origin": "Bushwick", "destination": "Wall St"}},
    {"id": "fr05", "category": "fresh_status",
     "question": "Q train ok?",
     "expect": {"kind": "line_status", "line": "Q"}},
    {"id": "fr06", "category": "fresh_status",
     "question": "are the A trains running normal today",
     "expect": {"kind": "line_status", "line": "A"}},
    {"id": "fr07", "category": "fresh_status",
     "question": "any service changes on the F this weekend?",
     "expect": {"kind": "line_status", "line": "F"}},
    {"id": "fr08", "category": "fresh_status",
     "question": "what's the deal with the 7 today",
     "expect": {"kind": "line_status", "line": "7"}},
    {"id": "fr09", "category": "fresh_info",
     "question": "is there an elevator outage at Grand Central",
     "expect": {"kind": "elevator"}},
    {"id": "fr10", "category": "fresh_info",
     "question": "closest train to the Empire State Building",
     "expect": {"kind": "geocode", "place": "Empire State Building"}},
    {"id": "fr11", "category": "fresh_info",
     "question": "show me the trains at Atlantic Av",
     "expect": {"kind": "station_info", "station": "Atlantic Av",
                "mentions_any": AMBIGUITY_WORDS}},
    {"id": "fr12", "category": "fresh_failure",
     "question": "14 St to 14 St",
     "expect": {"kind": "clarify", "term": "14 St"}},
    {"id": "fr13", "category": "fresh_failure",
     "question": "can you book me a taxi",
     "expect": {"kind": "out_of_scope"}},
    {"id": "fr14", "category": "fresh_failure",
     "question": "get me to Wall St",
     "expect": {"kind": "needs_origin"}},
    {"id": "fr15", "category": "fresh_failure",
     "question": "from Narnia to Times Square",
     "expect": {"kind": "graceful_fail"}},
]


# ── wild (v3 held-out) ─────────────────────────────────────────────────
# `fresh` scored 11/15, exposed four more parsing gaps, and those were fixed —
# so it is tuned-on now too, and these 15 were written afterwards and scored
# once. Each generation of this set costs less than the last (11/15 → …), which
# is itself the signal worth watching: when a new set stops finding anything,
# the parser has stopped being the bottleneck.
WILD_CASES: list[dict] = [
    {"id": "wd01", "category": "wild_route",
     "question": "how do i go from Rockefeller Center to Brooklyn Bridge",
     "expect": {"kind": "route", "origin": "Rockefeller Center", "destination": "Brooklyn Bridge"}},
    {"id": "wd02", "category": "wild_route",
     "question": "whats the ride time Union Sq to Astoria Blvd",
     "expect": {"kind": "route", "origin": "Union Sq", "destination": "Astoria Blvd"}},
    {"id": "wd03", "category": "wild_route",
     "question": "im near Columbus Circle, how do i reach Wall St",
     "expect": {"kind": "route", "origin": "Columbus Circle", "destination": "Wall St"}},
    {"id": "wd04", "category": "wild_route",
     "question": "can you route me: Marcy Av to Canal St",
     "expect": {"kind": "route", "origin": "Marcy Av", "destination": "Canal St"}},
    {"id": "wd05", "category": "wild_status",
     "question": "N train problems?",
     "expect": {"kind": "line_status", "line": "N"}},
    {"id": "wd06", "category": "wild_status",
     "question": "hows the L looking",
     "expect": {"kind": "line_status", "line": "L"}},
    {"id": "wd07", "category": "wild_status",
     "question": "is service normal on the E",
     "expect": {"kind": "line_status", "line": "E"}},
    {"id": "wd08", "category": "wild_status",
     "question": "anything wrong with the subway right now",
     "expect": {"kind": "service_status"}},
    {"id": "wd09", "category": "wild_info",
     "question": "what stops at Broadway Junction",
     "expect": {"kind": "station_info", "station": "Broadway Junction"}},
    {"id": "wd10", "category": "wild_info",
     "question": "nearest subway to Prospect Park",
     "expect": {"kind": "geocode", "place": "Prospect Park"}},
    {"id": "wd11", "category": "wild_info",
     "question": "elevators working at Jay St?",
     "expect": {"kind": "elevator"}},
    {"id": "wd12", "category": "wild_failure",
     "question": "need to get from Coney Island to JFK Airport",
     "expect": {"kind": "clarify", "term": "JFK Airport"}},
    {"id": "wd13", "category": "wild_failure",
     "question": "trip from 72 St to 72 St",
     "expect": {"kind": "clarify", "term": "72 St"}},
    {"id": "wd14", "category": "wild_failure",
     "question": "how do i get to Hogwarts from Times Square",
     "expect": {"kind": "graceful_fail"}},
    {"id": "wd15", "category": "wild_failure",
     "question": "whats the fare",
     "expect": {"kind": "out_of_scope"}},
]


SETS = {"core": CORE_CASES, "heldout": HELDOUT_CASES, "fresh": FRESH_CASES,
        "wild": WILD_CASES,
        "all": CORE_CASES + HELDOUT_CASES + FRESH_CASES + WILD_CASES}

CATEGORIES = tuple(dict.fromkeys(c["category"] for c in SETS["all"]))


def cases_for(which: str = "core", category: str | None = None) -> list[dict]:
    cases = list(SETS[which])
    if category in (None, "all"):
        return cases
    return [c for c in cases if c["category"] == category]
