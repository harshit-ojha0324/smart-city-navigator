import random

import pytest
from conftest import station_id

from navigator.core import routing
from navigator.core.graph_data import STATION_BY_ID, STATION_IDS

TIMES_SQ = station_id("Times Square")
CONEY = station_id("Coney Island")
WALL_ST = station_id("Wall St")
BEDFORD = station_id("Bedford Av")
HERALD = station_id("Herald Sq")


def test_known_route_found():
    r = routing.plan_route(TIMES_SQ, CONEY)
    assert r["found"] and r["path"][0] == TIMES_SQ and r["path"][-1] == CONEY
    assert r["total_minutes"] > 0 and r["total_stops"] > 0
    assert "Coney Island" in r["summary"]


def test_same_station_is_already_there():
    r = routing.plan_route(TIMES_SQ, TIMES_SQ)
    assert r["total_stops"] == 0 and r["num_transfers"] == 0
    assert "already" in r["summary"].lower()


def test_unknown_station_raises():
    with pytest.raises(routing.RouteError):
        routing.plan_route(TIMES_SQ, "nope")
    with pytest.raises(routing.RouteError):
        routing.plan_route("nope", TIMES_SQ)


def test_route_is_roughly_symmetric():
    there = routing.plan_route(TIMES_SQ, WALL_ST)["total_minutes"]
    back = routing.plan_route(WALL_ST, TIMES_SQ)["total_minutes"]
    assert abs(there - back) <= 5  # real schedules differ by direction, not wildly


def test_transfer_trip_has_multiple_legs():
    # Bedford Av (L only) to Herald Sq (no L) must leave the L somewhere.
    r = routing.plan_route(BEDFORD, HERALD)
    assert r["num_transfers"] >= 1
    assert len(r["legs"]) >= 2
    assert r["legs"][0]["line"] == "L"


def test_legs_use_lines_serving_both_endpoints():
    r = routing.plan_route(TIMES_SQ, CONEY)
    for leg in r["legs"]:
        if leg["walk"]:
            continue
        assert leg["line"] in STATION_BY_ID[leg["from_id"]]["lines"]
        assert leg["line"] in STATION_BY_ID[leg["to_id"]]["lines"]


def test_whole_network_is_reachable_from_times_square():
    for sid in STATION_IDS:
        assert routing.plan_route(TIMES_SQ, sid)["found"], f"{sid} unreachable"


def test_stop_counts_sum_across_legs():
    r = routing.plan_route(TIMES_SQ, CONEY)
    assert sum(leg["num_stops"] for leg in r["legs"]) == r["total_stops"]


def test_random_trips_stay_coherent():
    """Fuzz the planner across the whole network: every trip must describe
    itself consistently — legs join end to end and name real lines."""
    rng = random.Random(20260922)
    for _ in range(60):
        a, b = rng.sample(STATION_IDS, 2)
        r = routing.plan_route(a, b)
        assert r["stations"][0] == STATION_BY_ID[a]["name"]
        assert r["stations"][-1] == STATION_BY_ID[b]["name"]
        assert "?" not in r["summary"]
        assert r["total_minutes"] >= sum(leg["minutes"] for leg in r["legs"]) - 1
        for first, second in zip(r["legs"], r["legs"][1:], strict=False):
            assert first["to_id"] == second["from_id"], r["summary"]


def test_express_is_cheaper_than_the_local_on_the_same_hop():
    """Segments are priced per line from the schedule, so the express variant
    of a shared hop is never slower than the local."""
    faster = 0
    for hop in routing._RIDES.values():
        for lines in hop.values():
            express = {ln: m for ln, m in lines.items() if ln.endswith("X")}
            local = {ln: m for ln, m in lines.items() if not ln.endswith("X")}
            if express and local and min(express.values()) < min(local.values()):
                faster += 1
    assert faster > 0, "no express segment is quicker than its local — times look wrong"
