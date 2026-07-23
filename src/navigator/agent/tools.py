"""
Tool loading for the agent, in two interchangeable transports:

  * "mcp"       — connect to the 3 FastMCP servers over streamable HTTP via
                  langchain-mcp-adapters (the production path the gateway uses).
  * "inprocess" — LangChain tools with identical names that call the same core,
                  skipping the network hop (fast, hermetic path for tests/CI).

Both expose the same nine tool names, so the graph and the deterministic planner
are transport-agnostic.
"""
from __future__ import annotations

import json

from langchain_core.tools import tool

from navigator.core import geocode, mta_feed, routing
from navigator.core.graph_data import STATIONS
from navigator.mcp_servers import all_server_urls


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)


# ── In-process tools (same names/args as the MCP servers) ──────────────
@tool
def get_service_status() -> str:
    """Current service status for every NYC subway line (severity 0-3 per line)."""
    alerts = mta_feed.fetch_alerts()
    return _json({"summary": mta_feed.status_summary(alerts), "lines": alerts})


@tool
def get_line_status(line: str) -> str:
    """Service status for a single subway line (e.g. "A", "7", "Q")."""
    return _json(mta_feed.get_line_status(line))


@tool
def list_elevator_outages(station_contains: str = "") -> str:
    """Current elevator/escalator outages, optionally filtered by station substring."""
    outages = mta_feed.fetch_elevator_outages()
    if station_contains:
        needle = station_contains.lower()
        outages = [o for o in outages if needle in o.get("station", "").lower()]
    return _json({"count": len(outages), "outages": outages})


@tool
def plan_trip(origin: str, destination: str) -> str:
    """Plan the fastest subway trip between two free-text places.

    Returns numbered legs, total minutes, stops and transfers — or
    needs_disambiguation with candidates when an endpoint is ambiguous.
    """
    try:
        o = geocode.resolve_station(origin)
        d = geocode.resolve_station(destination)
    except geocode.GeocodeError as exc:
        return _json({"error": str(exc), "found": False})

    def pack(res):
        return {"id": res["station"]["id"], "name": res["station"]["name"],
                "ambiguous": res["ambiguous"],
                "candidates": [{"id": s, "name": geocode.STATION_BY_ID[s]["name"]}
                               for s in res["candidates"]]}

    op, dp = pack(o), pack(d)
    if op["ambiguous"] or dp["ambiguous"]:
        return _json({"needs_disambiguation": True, "origin": op, "destination": dp,
                      "message": "One or both endpoints matched multiple stations."})
    try:
        route = routing.plan_route(op["id"], dp["id"])
    except routing.RouteError as exc:
        return _json({"error": str(exc), "found": False})
    route["origin_station"] = op["name"]
    route["destination_station"] = dp["name"]
    return _json(route)


@tool
def plan_trip_by_id(origin_id: str, destination_id: str) -> str:
    """Plan a trip between two known station ids (skips name resolution)."""
    try:
        return _json(routing.plan_route(origin_id, destination_id))
    except routing.RouteError as exc:
        return _json({"error": str(exc), "found": False})


@tool
def list_stations(on_line: str = "") -> str:
    """List modeled stations, optionally only those served by a given line."""
    stations = STATIONS
    if on_line:
        line = on_line.strip().upper()
        stations = [s for s in stations if line in s["lines"]]
    return _json({"count": len(stations),
                  "stations": [{"id": s["id"], "name": s["name"], "lines": s["lines"]}
                               for s in stations]})


@tool
def geocode_place(query: str) -> str:
    """Resolve a free-text place/landmark to coordinates and the nearest station."""
    try:
        return _json(geocode.geocode_place(query))
    except geocode.GeocodeError as exc:
        return _json({"error": str(exc)})


@tool
def resolve_station(query: str) -> str:
    """Resolve free text to the best-matching station (+ ambiguity candidates)."""
    try:
        res = geocode.resolve_station(query)
        return _json({"station": res["station"], "ambiguous": res["ambiguous"],
                      "score": res["score"],
                      "candidates": [{"id": s, "name": geocode.STATION_BY_ID[s]["name"]}
                                     for s in res["candidates"]]})
    except geocode.GeocodeError as exc:
        return _json({"error": str(exc)})


@tool
def nearest_station(lat: float, lng: float) -> str:
    """Nearest modeled subway station to a latitude/longitude."""
    return _json(geocode.nearest_station(lat, lng))


INPROCESS_TOOLS = [
    get_service_status, get_line_status, list_elevator_outages,
    plan_trip, plan_trip_by_id, list_stations,
    geocode_place, resolve_station, nearest_station,
]


def inprocess_tools() -> list:
    return list(INPROCESS_TOOLS)


async def load_mcp_tools(host: str | None = None) -> list:
    """Load tools from the 3 live FastMCP servers over streamable HTTP."""
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(all_server_urls(host))
    return await client.get_tools()


async def build_tools(transport: str = "inprocess", host: str | None = None) -> list:
    if transport == "mcp":
        return await load_mcp_tools(host)
    return inprocess_tools()
