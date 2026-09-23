"""Regression tests for bugs found in the Sep 2026 audit — each test names the
failure it pins so a reintroduction reads clearly in CI."""
import asyncio
import os
import tempfile

import pytest
from conftest import station_id

from navigator.agent.graph import run_once
from navigator.agent.llm import DeterministicPlanner
from navigator.agent.nodes import message_text
from navigator.agent.tools import inprocess_tools
from navigator.core import geocode, mta_feed, routing
from navigator.core.graph_data import STATION_IDS
from navigator.gateway.app import create_app
from navigator.mcp_servers import alerts_server, geocode_server, routing_server


def _run(q, **kw):
    return asyncio.run(run_once(q, **kw))


# ── live feed correctness ──────────────────────────────────────────────
def _alert(route, alert_type, periods):
    return {"alert": {
        "active_period": periods,
        "informed_entity": [{"route_id": route}],
        "header_text": {"translation": [{"language": "en", "text": f"{route} {alert_type}"}]},
        "transit_realtime.mercury_alert": {"alert_type": alert_type},
    }}


def test_future_planned_work_is_not_reported_as_current():
    now = 1_800_000_000
    feed = {"entity": [
        _alert("C", "Planned - Suspended", [{"start": now + 3600, "end": now + 7200}]),  # tonight
        _alert("A", "Planned - Suspended", [{"start": now - 60, "end": now + 60}]),      # now
        _alert("L", "Reduced Service", []),                                               # open-ended
    ]}
    status = mta_feed._parse_alerts_json(feed, now_ts=now)
    assert status["C"]["severity"] == 0, "a suspension that starts later must not show as current"
    assert status["A"]["severity"] == 3
    assert status["L"]["severity"] == 1


def test_simulated_status_is_labelled_not_passed_off_as_live():
    a = _run("Is the L train running?")["answer"]
    assert "simulated" in a.lower()


def test_elevator_outages_never_claims_none_when_feed_unavailable():
    out = alerts_server.list_elevator_outages("Times Sq")
    assert out["available"] is False
    a = _run("Are there elevator outages at Times Sq?")["answer"]
    assert "unavailable" in a.lower() and "no elevator" not in a.lower()


def test_elevator_filter_matches_normalized_station_names(monkeypatch):
    monkeypatch.setattr(mta_feed, "fetch_elevator_outages", lambda: [
        {"station": "Times Sq-42 St", "type": "Elevator", "serving": "street to mezzanine",
         "reason": "Repair", "eta": "10/01/2026"},
        {"station": "Jamaica-179 St", "type": "Escalator", "serving": "", "reason": "", "eta": ""},
    ])
    out = alerts_server.list_elevator_outages("Times Square")
    assert out["available"] and out["count"] == 1


# ── checkpointed threads ───────────────────────────────────────────────
def test_reused_thread_does_not_replay_previous_answer():
    path = tempfile.mktemp(suffix=".sqlite")
    try:
        first = _run("How do I get from Times Square to Coney Island?",
                     thread_id="t1", checkpoint_path=path)
        second = _run("Is the L train running?", thread_id="t1", checkpoint_path=path)
        assert "Coney Island" in first["answer"]
        assert "Coney Island" not in second["answer"]
        assert [w["agent"] for w in second["worker_results"]] == ["service_advisor"]
        # ...while the thread still remembers the conversation.
        questions = [m.content for m in second["messages"] if m.type == "human"]
        assert questions == ["How do I get from Times Square to Coney Island?",
                             "Is the L train running?"]
    finally:
        os.path.exists(path) and os.remove(path)


# ── planner coverage ───────────────────────────────────────────────────
@pytest.mark.parametrize("q,line", [
    ("What's the status of the N?", "N"),
    ("Is the Q running?", "Q"),
    ("Is the F delayed?", "F"),
    ("Are there delays on the 7?", "7"),
])
def test_bare_line_letters_are_recognised(q, line):
    assert DeterministicPlanner()._extract_line(q) == line
    assert _run(q)["answer"].startswith(f"The {line} train")


def test_prose_never_reads_as_a_line():
    assert DeterministicPlanner()._extract_line("What is the status right now?") is None


def test_destination_only_asks_for_origin():
    st = _run("How do I get to Coney Island?")
    assert st["intent"] == "route"
    assert "where are you starting" in st["answer"].lower()
    assert "Coney Island" in st["answer"]


def test_origin_stated_up_front():
    a = _run("I'm at Union Square, how do I get to Harlem?")["answer"]
    assert a.startswith("Take the") and "Union Sq" in a


@pytest.mark.parametrize("question,expected", [
    # Each phrasing was found by a held-out eval set, in the generation named.
    ("what's the deal with the 7 today", "status"),      # fresh: naming a line is a service question
    ("show me the trains at Atlantic Av", "info"),       # fresh: a lookup, not a service question
    ("hows the L looking", "status"),                    # wild
    ("what stops at Broadway Junction", "info"),         # wild
])
def test_phrasings_found_by_held_out_sets_are_classified(question, expected):
    assert DeterministicPlanner().classify(question) == expected


@pytest.mark.parametrize("question,origin,destination", [
    ("im at Grand Central and need to reach Barclays Center", "Grand Central", "Barclays Center"),
    ("im near Columbus Circle, how do i reach Wall St", "Columbus Circle", "Wall St"),
    ("whats the ride time Union Sq to Astoria Blvd", "Union Sq", "Astoria Blvd"),
    ("can you route me: Marcy Av to Canal St", "Marcy Av", "Canal St"),
    ("from Bedford Av to Herald Sq", "Bedford Av", "Herald Sq"),  # "be" must not eat "Bedford"
])
def test_trip_endpoints_survive_the_sentence_around_them(question, origin, destination):
    assert DeterministicPlanner()._extract_od(question) == (origin, destination)


def test_a_pronoun_is_never_treated_as_a_station():
    """"get me to Wall St" used to plan a trip starting from "me"."""
    planner = DeterministicPlanner()
    assert planner._extract_od("get me to Wall St") is None
    assert planner.destination_only("get me to Wall St") == "Wall St"
    assert "where are you starting" in _run("get me to Wall St")["answer"].lower()


# ── routing quality ────────────────────────────────────────────────────
def test_landmark_endpoints_snap_to_nearest_station_and_say_so():
    a = _run("What's the best route from Williamsburg to Barclays Center?")["answer"]
    assert a.startswith("Take the")
    assert "nearest modeled station" in a  # substitution is disclosed, not silent


def test_far_landmark_is_refused_not_routed():
    # No subway reaches LaGuardia; the nearest stop is 3 km away, so a route
    # would be a confident fiction.
    with pytest.raises(geocode.GeocodeError):
        geocode.resolve_endpoint("LaGuardia Airport")
    a = _run("How do I get from Times Square to LaGuardia Airport?")["answer"]
    assert "Take the" not in a and "too far" in a


def test_airport_the_subway_does_reach_offers_both_airtrain_stations():
    # Two stations feed the JFK AirTrain (E/J/Z at Sutphin Blvd, A at Howard
    # Beach) — a real choice, so the agent asks instead of picking for you.
    a = _run("How do I get from Times Square to JFK Airport?")["answer"]
    assert "Sutphin Blvd-Archer Av-JFK Airport" in a and "Howard Beach-JFK Airport" in a
    assert "which did you mean" in a.lower()


def test_transfer_penalty_prefers_fewer_changes():
    # Bedford Av to Times Sq is one change off the L, not a three-seat scramble
    # that shaves a minute of riding.
    r = routing.plan_route(station_id("Bedford Av"), station_id("Times Square"))
    assert r["num_transfers"] <= 1
    assert r["legs"][0]["line"] == "L"


def test_every_route_names_a_real_line_or_a_walk():
    times_sq = station_id("Times Square")
    for sid in STATION_IDS:
        r = routing.plan_route(times_sq, sid)
        assert "?" not in r["summary"], r["summary"]
        for leg in r["legs"]:
            assert leg["walk"] or leg["line"] in geocode.STATION_BY_ID[leg["from_id"]]["lines"]


def test_cached_route_is_not_mutated_by_callers():
    routing_server.plan_trip("Williamsburg", "Times Square")
    direct = routing.plan_route(station_id("Marcy Av"), station_id("Times Square"))
    assert "nearest modeled" not in direct["summary"]


# ── multi-agent hand-off ───────────────────────────────────────────────
def test_compound_query_scopes_status_to_the_route_lines():
    st = _run("How do I get from Bedford Av to Herald Sq and are there delays?")
    route, status = st["worker_results"]
    lines = [leg["line"] for leg in route["data"]["legs"] if not leg["walk"]]
    assert lines == ["L", "F"]  # the real trip: L to 14 St, F to Herald Sq
    # Service Advisor reports exactly the handed-off lines, not all 23.
    assert status["result"].startswith("Service on your route")
    assert "L:" in status["result"] and "F:" in status["result"]
    assert "7:" not in status["result"]


def test_express_legs_are_checked_under_their_rider_facing_line():
    """A leg on the "7X" must be handed to the alerts feed as the 7 — asking
    about "7X" returns "unknown line" and that leg vanishes from the report."""
    st = _run("How do I get from Columbus Circle to Flushing Main St and are there delays?")
    route, status = st["worker_results"]
    assert any(leg["line"].endswith("X") for leg in route["data"]["legs"])
    assert "7:" in status["result"]
    assert "7X" not in status["result"] and "unknown" not in status["result"].lower()


def test_the_itinerary_is_handed_on_even_if_another_tool_ran_last():
    from navigator.agent.workers import _carry_forward

    results = [("plan_trip", {"legs": [{"line": "L"}], "summary": "…"}),
               ("resolve_station", {"station": {"name": "14 St"}})]
    assert _carry_forward(results)["legs"] == [{"line": "L"}]


def test_tools_accept_the_argument_types_models_actually_send():
    """A local model called get_line_status(line=7) as an integer; the schema
    rejected it, and the agent burned its retries on the same bad call."""
    assert alerts_server.get_line_status(7)["line"] == "7"
    assert alerts_server.get_line_status("7")["line"] == "7"
    assert geocode_server.nearest_station("40.7580", "-73.9855")["name"] == "Times Sq-42 St"
    assert "error" in geocode_server.nearest_station("uptown", "somewhere")
    assert routing_server.list_stations(7)["count"] > 0
    assert alerts_server.list_elevator_outages(0)["available"] is False


# ── transports & gateway ───────────────────────────────────────────────
def test_inprocess_tools_mirror_mcp_server_schemas():
    async def mcp_tools():
        out = {}
        for server in (alerts_server, routing_server, geocode_server):
            for t in await server.mcp.list_tools():
                out[t.name] = set(t.input_schema.get("properties", {}))
        return out

    mcp = asyncio.run(mcp_tools())
    local = {t.name: set(t.args) for t in inprocess_tools()}
    assert local == mcp


def test_gemini_list_content_is_flattened_to_text():
    assert message_text([{"type": "text", "text": "Take "}, "the L"]) == "Take the L"
    assert message_text("plain") == "plain"


def test_gateway_gives_each_request_its_own_thread():
    c = create_app().test_client()
    a = c.post("/api/ask", json={"question": "Is the L train running?"}).get_json()
    b = c.post("/api/ask", json={"question": "Is the L train running?"}).get_json()
    assert a["thread_id"] != b["thread_id"]


def test_gateway_rejects_oversized_questions():
    c = create_app().test_client()
    assert c.post("/api/ask", json={"question": "x" * 501}).status_code == 413


def test_demo_page_never_injects_step_text_as_html():
    body = create_app().test_client().get("/").get_data(as_text=True)
    assert "innerHTML" not in body
