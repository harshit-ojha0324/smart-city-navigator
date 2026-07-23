#!/usr/bin/env python3
"""
CLI demo — ask the agent a question and watch it reason.

    python scripts/demo.py "How do I get from Times Square to Coney Island?"
    python scripts/demo.py --mcp "Is the L train running?"   # via live MCP servers

With no argument, drops into a small REPL. Uses the in-process transport by
default so it runs without booting the MCP servers.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

os.environ.setdefault("NAVIGATOR_SIMULATE_FEED", "1")

from navigator.agent.graph import stream_once  # noqa: E402

C = {"understand": "\033[36m", "plan": "\033[33m", "tools": "\033[35m",
     "synthesize": "\033[32m", "done": "\033[1;32m", "error": "\033[31m"}
R = "\033[0m"


async def ask(question: str, transport: str, thread_id: str = "cli") -> None:
    print(f"\n\033[1mQ:\033[0m {question}")
    async for ev in stream_once(question, thread_id=thread_id, transport=transport):
        node = ev.get("node", "")
        if node == "done":
            print(f"\n{C['done']}Answer:{R} {ev.get('answer','')}\n")
        else:
            color = C.get(node, "")
            print(f"  {color}[{node}]{R} {ev.get('text','')}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*", help="the transit question")
    ap.add_argument("--mcp", action="store_true", help="route through the live MCP servers")
    args = ap.parse_args()
    transport = "mcp" if args.mcp else "inprocess"

    if args.question:
        asyncio.run(ask(" ".join(args.question), transport))
        return

    print("Smart City Navigator — ask a transit question (Ctrl-D to quit)")
    n = 0
    while True:
        try:
            q = input("\n› ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q:
            n += 1
            asyncio.run(ask(q, transport, thread_id=f"cli-{n}"))


if __name__ == "__main__":
    main()
