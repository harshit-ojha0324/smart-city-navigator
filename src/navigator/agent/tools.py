"""
Tool loading for the agent, in two interchangeable transports:

  * "mcp"       — connect to the 3 FastMCP servers over streamable HTTP via
                  langchain-mcp-adapters (the production path the gateway uses).
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
    routing_server.plan_trip_by_id,
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


INPROCESS_TOOLS: list[BaseTool] = [_as_langchain_tool(fn) for fn in _SERVER_FUNCTIONS]


def inprocess_tools() -> list:
    return list(INPROCESS_TOOLS)


# Tool *definitions* are stable for a server's lifetime, and each MCP tool call
# opens its own session, so the listing can be reused across requests instead of
# re-running the 3-server handshake + list_tools on every question.
_MCP_TOOL_CACHE: dict[str | None, list] = {}


async def load_mcp_tools(host: str | None = None) -> list:
    """Load tools from the 3 live FastMCP servers over streamable HTTP."""
    if host not in _MCP_TOOL_CACHE:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient(all_server_urls(host))
        _MCP_TOOL_CACHE[host] = await client.get_tools()
    return list(_MCP_TOOL_CACHE[host])


async def build_tools(transport: str = "inprocess", host: str | None = None) -> list:
    if transport == "mcp":
        return await load_mcp_tools(host)
    return inprocess_tools()
