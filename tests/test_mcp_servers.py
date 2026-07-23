"""In-memory tests of the three FastMCP servers (no HTTP transport needed)."""
import asyncio
import json

from navigator.mcp_servers import alerts_server, geocode_server, routing_server


def call(server, name, args):
    res = asyncio.run(server.mcp.call_tool(name, args))
    # FastMCP returns [TextContent(text=<json>)] for dict/list tool returns.
    payload = res[0] if isinstance(res, (list, tuple)) else res
    text = getattr(payload, "text", None)
    return json.loads(text) if text else payload


def test_alerts_service_status():
    out = call(alerts_server, "get_service_status", {})
    assert "summary" in out and "lines" in out
    assert len(out["lines"]) == len(alerts_server.mta_feed.SUBWAY_LINE_IDS)


def test_alerts_line_status():
    out = call(alerts_server, "get_line_status", {"line": "A"})
    assert out["line"] == "A" and "severity" in out


def test_routing_plan_trip_success():
    out = call(routing_server, "plan_trip",
               {"origin": "Times Square", "destination": "Coney Island"})
    assert out.get("found") and out["num_transfers"] >= 0
    assert "Coney Island" in out["summary"]


def test_routing_plan_trip_ambiguous():
    out = call(routing_server, "plan_trip",
               {"origin": "125 St", "destination": "Times Square"})
    assert out.get("needs_disambiguation") is True


def test_routing_list_stations_on_line():
    out = call(routing_server, "list_stations", {"on_line": "L"})
    assert out["count"] >= 1
    assert all("L" in s["lines"] for s in out["stations"])


def test_geocode_place():
    out = call(geocode_server, "geocode_place", {"query": "Empire State Building"})
    assert out["nearest_station"]["id"] == "hz"


def test_geocode_resolve_ambiguous():
    out = call(geocode_server, "resolve_station", {"query": "125 St"})
    assert out["ambiguous"] is True


def test_geocode_nearest_station():
    out = call(geocode_server, "nearest_station", {"lat": 40.7580, "lng": -73.9855})
    assert out["id"] == "ts"


def test_all_servers_expose_expected_tools():
    async def names(server):
        return {t.name for t in await server.mcp.list_tools()}

    assert {"get_service_status", "get_line_status", "list_elevator_outages"} <= asyncio.run(names(alerts_server))
    assert {"plan_trip", "plan_trip_by_id", "list_stations"} <= asyncio.run(names(routing_server))
    assert {"geocode_place", "resolve_station", "nearest_station"} <= asyncio.run(names(geocode_server))
