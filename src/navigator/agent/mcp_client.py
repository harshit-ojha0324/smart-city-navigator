"""
MCP tools → LangChain tools, spoken directly to the servers.

This used to be `langchain-mcp-adapters`. That package still pins `mcp<2.0.0`,
so staying on it meant staying on the v1 SDK; talking to the servers ourselves
is about sixty lines and drops a dependency from the critical path.

What it does: connect to each server over streamable HTTP, read its tool list,
and wrap every tool as a LangChain `StructuredTool` whose arguments come from
the server's own JSON schema. Each call opens its own session — MCP sessions
are per-exchange here, so nothing has to be kept alive between questions, and
a server that restarts is picked up on the next call rather than poisoning a
cached connection.

Errors are raised as `ToolException` so LangGraph's ToolNode turns them into an
error ToolMessage the agent can read and react to, exactly as the adapter did.
"""
from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, StructuredTool, ToolException
from mcp.client.client import Client


def _text_of(result: Any) -> str:
    """Flatten an MCP tool result to the text the agent consumes."""
    parts = [getattr(block, "text", "") for block in (result.content or [])]
    return "\n".join(p for p in parts if p)


def _make_tool(url: str, name: str, description: str, schema: dict) -> BaseTool:
    async def call(**kwargs: Any) -> str:
        async with Client(url) as session:
            result = await session.call_tool(name, kwargs)
        text = _text_of(result)
        if getattr(result, "is_error", False):
            raise ToolException(text or f"{name} failed")
        return text

    return StructuredTool(
        name=name,
        description=description or name,
        args_schema=schema or {"type": "object", "properties": {}},
        coroutine=call,
        handle_tool_error=True,
    )


async def load_tools(server_urls: dict[str, str]) -> list[BaseTool]:
    """Every tool exposed by the given servers, in server order.

    `server_urls` maps a server name to its streamable-HTTP endpoint. Tool
    *definitions* are read once here; the calls themselves connect per use.
    """
    tools: list[BaseTool] = []
    for url in server_urls.values():
        async with Client(url) as session:
            listing = await session.list_tools()
        for tool in listing.tools:
            tools.append(_make_tool(url, tool.name, tool.description or "",
                                    dict(tool.input_schema or {})))
    return tools
