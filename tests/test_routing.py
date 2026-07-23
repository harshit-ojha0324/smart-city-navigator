import pytest

from navigator.core import routing
from navigator.core.graph_data import STATION_BY_ID, STATION_IDS


def test_known_route_found():
    r = routing.plan_route("ts", "ci")
    assert r["found"] and r["path"][0] == "ts" and r["path"][-1] == "ci"
    assert r["total_minutes"] > 0 and r["total_stops"] > 0
    assert "Coney Island" in r["summary"]


def test_same_station_is_already_there():
    r = routing.plan_route("ts", "ts")
    assert r["total_stops"] == 0 and r["num_transfers"] == 0
    assert "already" in r["summary"].lower()


def test_unknown_station_raises():
    with pytest.raises(routing.RouteError):
        routing.plan_route("ts", "nope")
    with pytest.raises(routing.RouteError):
        routing.plan_route("nope", "ts")


def test_route_is_roughly_symmetric():
    a = routing.plan_route("ts", "fc")["total_minutes"]
    b = routing.plan_route("fc", "ts")["total_minutes"]
    assert a == b  # undirected graph, min-weight edges


def test_transfer_trip_has_multiple_legs():
    # Bedford Av (L only) to Herald Sq (no L) must transfer off the L.
    r = routing.plan_route("bed", "hz")
    assert r["num_transfers"] >= 1
    assert len(r["legs"]) == r["num_transfers"] + 1


def test_legs_use_lines_serving_both_endpoints():
    r = routing.plan_route("ts", "ci")
    for leg in r["legs"]:
        line = leg["line"]
        assert line in STATION_BY_ID[leg["from_id"]]["lines"]
        assert line in STATION_BY_ID[leg["to_id"]]["lines"]


def test_graph_is_connected_from_times_square():
    # Every modeled station is reachable from Times Sq.
    for sid in STATION_IDS:
        r = routing.plan_route("ts", sid)
        assert r["found"], f"{sid} unreachable"


def test_stop_counts_sum_across_legs():
    r = routing.plan_route("ts", "ci")
    assert sum(leg["num_stops"] for leg in r["legs"]) == r["total_stops"]
