"""
Tool loading for the agent, in two interchangeable transports:

  * "mcp"       — connect to the 3 MCP servers over streamable HTTP (the
                  production path the gateway uses), via `mcp_client.py`.
  * "inprocess" — the *same* server tool functions wrapped as LangChain tools,
                  skipping the network hop (fast, hermetic path for tests/CI).

The in-process tools are generated from the MCP servers' own functions, so both
transports share one implementation, one set of names/args, and one docstring
per tool — they cannot drift apart. The graph and planner are transport-agnostic.
"""
from __future__ import annotations

import functools
import json

from langchain_core.tools import BaseTool, tool

from navigator.mcp_servers import alerts_server, all_server_urls, geocode_server, routing_server

# Every tool the three servers expose, in a stable order.
_SERVER_FUNCTIONS = [
    alerts_server.get_service_status,
    alerts_server.get_line_status,
    alerts_server.list_elevator_outages,
    routing_server.plan_trip,
    routing_server.list_stations,
    geocode_server.geocode_place,
    geocode_server.resolve_station,
    geocode_server.nearest_station,
]


def _as_langchain_tool(fn) -> BaseTool:
    """Wrap a server tool function; its dict result is JSON-encoded exactly as
    the MCP transport would deliver it as text content."""
    @functools.wraps(fn)
    def run(*args, **kwargs) -> str:
        return json.dumps(fn(*args, **kwargs), ensure_ascii=False)

    run.__annotations__ = {**fn.__annotations__, "return": str}
    return tool(run)


def inprocess_tools() -> list[BaseTool]:
    return [_as_langchain_tool(fn) for fn in _SERVER_FUNCTIONS]


# Tool *definitions* are stable for a server's lifetime, and each MCP tool call
# opens its own session, so the listing can be reused across requests instead of
# re-running the 3-server handshake + list_tools on every question.
_mcp_tools: list | None = None


async def load_mcp_tools() -> list:
    """Load tools from the 3 live MCP servers over streamable HTTP."""
    global _mcp_tools
    if _mcp_tools is None:
        from .mcp_client import load_tools

        _mcp_tools = await load_tools(all_server_urls())
    return list(_mcp_tools)


async def build_tools(transport: str = "inprocess") -> list:
    if transport == "mcp":
        return await load_mcp_tools()
    return inprocess_tools()
