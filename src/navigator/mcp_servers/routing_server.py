"""
Routing MCP server — Dijkstra trip planning over the 63-station subway graph.

Wraps navigator.core.routing (+ geocode for name resolution). When an endpoint
is ambiguous it returns needs_disambiguation with candidates instead of guessing,
letting the agent resolve or ask a follow-up.

Run standalone:  python -m navigator.mcp_servers.routing_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from navigator.core import geocode, routing
from navigator.core.graph_data import STATIONS
from navigator.mcp_servers import HOST, PORTS

mcp = FastMCP("navigator-routing", host=HOST, port=PORTS["routing"])


def _via_note(label: str, end: dict) -> str:
    via = end.get("via")
    if not via:
        return ""
    return (f"{label} {geocode.display_name(via['place'])} → nearest modeled station {end['name']} "
            f"({via['distance_km']} km away).")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Plan a subway trip (by place name)",
        readOnlyHint=True,
        openWorldHint=False,  # pure Dijkstra over the fixed in-repo station graph
    )
)
def plan_trip(origin: str | int, destination: str | int) -> dict:
    """Plan the fastest subway trip. Use this for any trip question.

    `origin` and `destination` are free text — a station name, a landmark, a
    neighborhood, or a station id returned by another tool. Resolves each
    endpoint to a station (snapping a landmark such as "Williamsburg" to its
    nearest station, noted in the summary), then runs Dijkstra. If either endpoint
    is ambiguous, returns {"needs_disambiguation": True, ...} with candidates so
    the caller can clarify. Otherwise returns a full itinerary: numbered legs,
    total minutes, stop count, transfer count, and a summary sentence.
    """
    try:
        o = geocode.resolve_endpoint(origin)
        d = geocode.resolve_endpoint(destination)
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
    route = dict(route)  # plan_route is lru_cached — never mutate the shared dict
    route["origin_station"] = o["name"]
    route["destination_station"] = d["name"]
    notes = " ".join(n for n in (_via_note("From", o), _via_note("To", d)) if n)
    if notes:
        route["summary"] = f"{route['summary']} ({notes})"
    return route


@mcp.tool(
    annotations=ToolAnnotations(
        title="Plan a subway trip (by station id)",
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def plan_trip_by_id(origin_id: str | int, destination_id: str | int) -> dict:
    """Plan a trip between two known station ids. Prefer `plan_trip`, which
    takes ids as well as names; this exists for callers that already have ids."""
    try:
        return routing.plan_route(str(origin_id), str(destination_id))
    except routing.RouteError as exc:
        return {"error": str(exc), "found": False}


@mcp.tool(
    annotations=ToolAnnotations(
        title="List modeled stations",
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def list_stations(on_line: str | int = "", limit: int = 50) -> dict:
    """List modeled stations, optionally only those served by a given line.

    The network has 475 stations, so results are capped (default 50) — an
    unfiltered dump is ~28 KB of context for no benefit. `count` is the true
    number of matches; narrow with `on_line` rather than raising the limit.
    """
    stations = STATIONS
    if on_line:
        line = str(on_line).strip().upper()
        stations = [s for s in stations if line in geocode.rider_lines(s["lines"])]
    limit = max(1, min(int(limit or 50), 200))
    shown = stations[:limit]
    return {
        "count": len(stations),
        "returned": len(shown),
        "truncated": len(shown) < len(stations),
        "stations": [
            {"id": s["id"], "name": s["name"], "lines": geocode.rider_lines(s["lines"])}
            for s in shown
        ],
    }


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
