"""
Geocode MCP server — free-text place / station resolution over streamable HTTP.

Wraps navigator.core.geocode: landmark and station-name lookup plus
nearest-station-to-coordinate. Deterministic and offline, so tool-calls are
reproducible in the eval suite.

Run standalone:  python -m navigator.mcp_servers.geocode_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from navigator.core import geocode
from navigator.mcp_servers import HOST, PORTS

mcp = FastMCP("navigator-geocode", host=HOST, port=PORTS["geocode"])


@mcp.tool(
    annotations=ToolAnnotations(
        title="Geocode a place/landmark",
        readOnlyHint=True,
        openWorldHint=False,  # deterministic offline landmark + station tables
    )
)
def geocode_place(query: str) -> dict:
    """Resolve a free-text place or landmark to coordinates and the nearest station.

    Tries a landmark table first (e.g. "Empire State Building"), then station
    names. Returns {lat, lng, source, matched, nearest_station{...}}.
    """
    try:
        return geocode.geocode_place(query)
    except geocode.GeocodeError as exc:
        return {"error": str(exc)}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Resolve text to a station",
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def resolve_station(query: str) -> dict:
    """Resolve free text to the best-matching subway station.

    Returns the matched station, a candidate list, and an `ambiguous` flag when
    several stations share the name (e.g. "125 St").
    """
    try:
        res = geocode.resolve_station(query)
        return {
            "station": res["station"],
            "candidates": [
                {"id": sid, "name": geocode.STATION_BY_ID[sid]["name"]}
                for sid in res["candidates"]
            ],
            "ambiguous": res["ambiguous"],
            "score": res["score"],
        }
    except geocode.GeocodeError as exc:
        return {"error": str(exc)}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Nearest station to coordinates",
        readOnlyHint=True,
        openWorldHint=False,
    )
)
def nearest_station(lat: float, lng: float) -> dict:
    """Nearest modeled subway station to a latitude/longitude, with distance_km."""
    return geocode.nearest_station(lat, lng)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
