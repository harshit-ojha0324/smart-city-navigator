"""End-to-end over the real transport: boot the 3 FastMCP servers as separate
processes speaking streamable HTTP, load their tools with langchain-mcp-adapters,
and answer questions through the full supervisor graph.

The other tests use the in-process transport; this one proves the resume's
"3 MCP servers over streamable HTTP" path actually works, on every CI run.
"""
import asyncio
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

import navigator.mcp_servers as servers
from navigator.agent import tools as agent_tools
from navigator.agent.graph import run_once

SRC = Path(__file__).resolve().parent.parent / "src"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            proc.log.seek(0)
            raise RuntimeError(f"MCP server on :{port} exited early:\n{proc.log.read()}")
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return
        except OSError:
            time.sleep(0.1)
    raise TimeoutError(f"MCP server never listened on :{port}")


@pytest.fixture(scope="module")
def live_mcp_servers():
    ports = {name: _free_port() for name in servers.PORTS}
    env = {**os.environ, "PYTHONPATH": str(SRC), "NAVIGATOR_SIMULATE_FEED": "1",
           "NAVIGATOR_MCP_HOST": "127.0.0.1",
           **{f"NAVIGATOR_{n.upper()}_PORT": str(p) for n, p in ports.items()}}
    procs = []
    for name in ports:
        # Logs go to a file, not a pipe: an undrained pipe fills and blocks the server.
        log = tempfile.TemporaryFile(mode="w+")
        proc = subprocess.Popen([sys.executable, "-m", f"navigator.mcp_servers.{name}_server"],
                                env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        proc.log = log
        procs.append(proc)
    saved = dict(servers.PORTS)
    try:
        for port, proc in zip(ports.values(), procs, strict=True):
            _wait_for(port, proc)
        servers.PORTS.update(ports)  # point the client at this test's servers
        agent_tools._MCP_TOOL_CACHE.clear()
        yield ports
    finally:
        servers.PORTS.update(saved)
        agent_tools._MCP_TOOL_CACHE.clear()
        for proc in procs:
            proc.terminate()
        for proc in procs:
            proc.wait(timeout=10)
            proc.log.close()


def test_tools_load_over_streamable_http(live_mcp_servers):
    tools = asyncio.run(agent_tools.load_mcp_tools("127.0.0.1"))
    assert {t.name for t in tools} == {t.name for t in agent_tools.inprocess_tools()}


@pytest.mark.parametrize("question,expect", [
    ("How do I get from Times Square to Coney Island?", "Take the N"),
    ("Is the L train running?", "The L train"),
    ("Which lines stop at Atlantic Av?", "Atlantic Av"),
    ("How do I get from Bedford Av to Herald Sq and are there delays?", "Service on your route"),
])
def test_agent_answers_through_live_mcp_servers(live_mcp_servers, question, expect):
    st = asyncio.run(run_once(question, transport="mcp", host="127.0.0.1"))
    assert expect in st["answer"]
