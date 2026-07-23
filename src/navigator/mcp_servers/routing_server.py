"""
Routing MCP server — Dijkstra trip planning over the 63-station subway graph.

Wraps navigator.core.routing (+ geocode for name resolution). When an endpoint
is ambiguous it returns needs_disambiguation with candidates instead of guessing,
letting the agent resolve or ask a follow-up.

Run standalone:  python -m navigator.mcp_servers.routing_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from navigator.core import geocode, routing
from navigator.core.graph_data import STATIONS
from navigator.mcp_servers import HOST, PORTS

mcp = FastMCP("navigator-routing", host=HOST, port=PORTS["routing"])


def _resolve(text: str) -> dict:
    """Resolve free text to a station, surfacing ambiguity as data (not an error)."""
    res = geocode.resolve_station(text)
    return {
        "id": res["station"]["id"],
        "name": res["station"]["name"],
        "ambiguous": res["ambiguous"],
        "candidates": [
            {"id": sid, "name": geocode.STATION_BY_ID[sid]["name"]}
            for sid in res["candidates"]
        ],
    }


@mcp.tool()
def plan_trip(origin: str, destination: str) -> dict:
    """Plan the fastest subway trip between two places given as free text.

    Resolves each endpoint to a station, then runs Dijkstra. If either endpoint
    is ambiguous, returns {"needs_disambiguation": True, ...} with candidates so
    the caller can clarify. Otherwise returns a full itinerary: numbered legs,
    total minutes, stop count, transfer count, and a summary sentence.
    """
    try:
        o = _resolve(origin)
        d = _resolve(destination)
    except geocode.GeocodeError as exc:
        return {"error": str(exc), "found": False}

    if o["ambiguous"] or d["ambiguous"]:
        return {
            "needs_disambiguation": True,
            "origin": o,
            "destination": d,
            "message": "One or both endpoints matched multiple stations; "
                       "confirm which station is meant.",
        }

    try:
        route = routing.plan_route(o["id"], d["id"])
    except routing.RouteError as exc:
        return {"error": str(exc), "found": False}
    route["origin_station"] = o["name"]
    route["destination_station"] = d["name"]
    return route


@mcp.tool()
def plan_trip_by_id(origin_id: str, destination_id: str) -> dict:
    """Plan a trip between two known station ids (skips name resolution)."""
    try:
        return routing.plan_route(origin_id, destination_id)
    except routing.RouteError as exc:
        return {"error": str(exc), "found": False}


@mcp.tool()
def list_stations(on_line: str = "") -> dict:
    """List modeled stations, optionally only those served by a given line."""
    stations = STATIONS
    if on_line:
        line = on_line.strip().upper()
        stations = [s for s in stations if line in s["lines"]]
    return {
        "count": len(stations),
        "stations": [
            {"id": s["id"], "name": s["name"], "lines": s["lines"]} for s in stations
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
