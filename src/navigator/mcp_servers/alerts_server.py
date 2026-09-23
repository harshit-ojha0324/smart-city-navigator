"""
Alerts MCP server — live NYC subway service status over streamable HTTP.

Wraps navigator.core.mta_feed. Every tool degrades to the deterministic
simulation on an upstream failure, so the agent always gets an answer.

Run standalone:  python -m navigator.mcp_servers.alerts_server
"""
from __future__ import annotations

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

from navigator.core import geocode, mta_feed
from navigator.mcp_servers import HOST, PORTS

mcp = MCPServer("navigator-alerts")


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get subway service status",
        readOnlyHint=True,
        openWorldHint=True,  # reads the live MTA feed — data changes outside our control
    )
)
def get_service_status() -> dict:
    """Current service status for every NYC subway line.

    Returns a summary string plus a per-line map of
    {severity 0-3, status, message, updatedAt}. Severity: 0 good, 1 delays,
    2 service change/planned work, 3 suspended.
    """
    alerts = mta_feed.fetch_alerts()
    return {"summary": mta_feed.status_summary(alerts), "lines": alerts}


@mcp.tool(
    annotations=ToolAnnotations(
        title="Get single line status",
        readOnlyHint=True,
        openWorldHint=True,
    )
)
def get_line_status(line: str | int) -> dict:
    """Service status for a single subway line (e.g. "A", "7", "Q").

    Accepts a number as well as a string: models routinely send the 7 train as
    the integer 7, and a schema that rejects it just burns retries.
    """
    return mta_feed.get_line_status(str(line))


@mcp.tool(
    annotations=ToolAnnotations(
        title="List elevator/escalator outages",
        readOnlyHint=True,
        openWorldHint=True,
    )
)
def list_elevator_outages(station_contains: str | int = "") -> dict:
    """Current elevator/escalator outages, optionally filtered by station name.

    Matching is normalized ("Times Square" finds "Times Sq-42 St"). Returns
    {"available": False} when the live outage feed can't be reached, so the
    caller never reports "no outages" it didn't actually observe.
    """
    station_contains = str(station_contains or "")
    outages = mta_feed.fetch_elevator_outages()
    if outages is None:
        return {"available": False, "count": 0, "outages": [], "station": station_contains,
                "message": "The live elevator/escalator outage feed is unavailable right now."}
    if station_contains:
        needle = geocode.normalize(station_contains)
        outages = [o for o in outages if needle and needle in geocode.normalize(o.get("station", ""))]
    return {"available": True, "count": len(outages), "outages": outages,
            "station": station_contains}


if __name__ == "__main__":
    # mcp 2.x moved the bind address off the constructor and onto run().
    mcp.run(transport="streamable-http", host=HOST, port=PORTS["alerts"])
