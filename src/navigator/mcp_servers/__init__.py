"""
Three MCP tool servers, each wrapping one slice of the transit core and speaking
the Model Context Protocol over streamable HTTP (the `mcp` SDK's MCPServer).

    alerts   — live GTFS-RT service status + elevator outages
    routing  — transfer-aware Dijkstra over the 475-station graph
    geocode  — free-text place / station resolution

The agent connects to all three through `agent/mcp_client.py`, so the LLM acts
on real-time MTA data through a standardized protocol rather than bespoke calls.
"""
from __future__ import annotations

import os

HOST = os.environ.get("NAVIGATOR_MCP_HOST", "127.0.0.1")

PORTS = {
    "alerts": int(os.environ.get("NAVIGATOR_ALERTS_PORT", "8071")),
    "routing": int(os.environ.get("NAVIGATOR_ROUTING_PORT", "8072")),
    "geocode": int(os.environ.get("NAVIGATOR_GEOCODE_PORT", "8073")),
}


def _host_for(name: str) -> str:
    """Per-server host override (e.g. Docker service names), else the default."""
    return os.environ.get(f"NAVIGATOR_{name.upper()}_HOST") or HOST


def server_url(name: str, host: str | None = None) -> str:
    """Streamable-HTTP endpoint URL for one server (mounted at /mcp)."""
    return f"http://{host or _host_for(name)}:{PORTS[name]}/mcp"


def all_server_urls(host: str | None = None) -> dict[str, dict]:
    """Per-server endpoint config consumed by `agent/mcp_client.py`."""
    return {
        name: {"url": server_url(name, host), "transport": "streamable_http"}
        for name in PORTS
    }
