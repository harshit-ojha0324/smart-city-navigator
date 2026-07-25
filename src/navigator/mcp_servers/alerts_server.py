"""
Alerts MCP server — live NYC subway service status over streamable HTTP.

Wraps navigator.core.mta_feed. Every tool degrades to the deterministic
simulation on an upstream failure, so the agent always gets an answer.

Run standalone:  python -m navigator.mcp_servers.alerts_server
"""
from __future__ import annotations

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from navigator.core import mta_feed
from navigator.mcp_servers import HOST, PORTS

mcp = FastMCP("navigator-alerts", host=HOST, port=PORTS["alerts"])


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
def get_line_status(line: str) -> dict:
    """Service status for a single subway line (e.g. "A", "7", "Q")."""
    return mta_feed.get_line_status(line)


@mcp.tool(
    annotations=ToolAnnotations(
        title="List elevator/escalator outages",
        readOnlyHint=True,
        openWorldHint=True,
    )
)
def list_elevator_outages(station_contains: str = "") -> dict:
    """Current elevator/escalator outages, optionally filtered by station name substring."""
    outages = mta_feed.fetch_elevator_outages()
    if station_contains:
        needle = station_contains.lower()
        outages = [o for o in outages if needle in o.get("station", "").lower()]
    return {"count": len(outages), "outages": outages}


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
