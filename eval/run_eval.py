#!/usr/bin/env python3
"""
Run the 20-prompt eval suite against the agent and report pass rates.

    python eval/run_eval.py                     # in-process, feed stubbed
    python eval/run_eval.py --transport mcp     # through the live MCP servers
    python eval/run_eval.py --category ambiguous
    python eval/run_eval.py --langsmith         # also log a scored LangSmith run

LangSmith tracing turns on automatically when LANGSMITH_API_KEY (or
LANGCHAIN_API_KEY) is set — every node/tool call is then traceable in the UI.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

from dataset import CATEGORIES, cases_for  # noqa: E402
from graders import grade  # noqa: E402


def _enable_langsmith_tracing() -> bool:
    key = os.environ.get("LANGSMITH_API_KEY") or os.environ.get("LANGCHAIN_API_KEY")
    if not key:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "smart-city-navigator-eval")
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    return True


async def _run_all(cases, transport):
    from navigator.agent.graph import run_once

    results = []
    for i, case in enumerate(cases):
        state = await run_once(case["question"], thread_id=f"eval-{case['id']}",
                               transport=transport)
        passed, reason = grade(case, state)
        results.append((case, state, passed, reason))
    return results


def _print_report(results) -> float:
    print(f"\n{'ID':<6}{'CATEGORY':<14}{'RESULT':<7} REASON")
    print("-" * 78)
    by_cat: dict[str, list[bool]] = {}
    for case, state, passed, reason in results:
        by_cat.setdefault(case["category"], []).append(passed)
        mark = "PASS" if passed else "FAIL"
        print(f"{case['id']:<6}{case['category']:<14}{mark:<7} {reason}")
        if not passed:
            print(f"        Q: {case['question']}")
            print(f"        A: {state.get('answer','')[:100]}")

    print("-" * 78)
    total = sum(1 for *_x, p, _ in results if p)
    for cat in CATEGORIES:
        flags = by_cat.get(cat, [])
        if flags:
            print(f"  {cat:<14} {sum(flags)}/{len(flags)}")
    rate = total / len(results) if results else 0.0
    print(f"\n  OVERALL: {total}/{len(results)}  ({rate:.0%})")
    return rate


def _maybe_langsmith(cases, transport):
    """Create/refresh a LangSmith dataset and run a scored experiment."""
    try:
        from langsmith import Client, evaluate
    except Exception as exc:
        print(f"[langsmith] SDK unavailable: {exc}")
        return
    from navigator.agent.graph import run_once
    from graders import langsmith_correctness

    client = Client()
    ds_name = "smart-city-navigator-20"
    if not client.has_dataset(dataset_name=ds_name):
        ds = client.create_dataset(ds_name, description="Smart City Navigator eval set")
        client.create_examples(
            inputs=[{"question": c["question"]} for c in cases],
            outputs=[c["expect"] for c in cases],
            metadata=[{"case": c} for c in cases],
            dataset_id=ds.id,
        )

    def target(inputs: dict) -> dict:
        state = asyncio.run(run_once(inputs["question"], transport=transport))
        return {"answer": state.get("answer", ""), "intent": state.get("intent", ""),
                "state": state}

    evaluate(target, data=ds_name, evaluators=[langsmith_correctness],
             experiment_prefix="scn-eval")
    print(f"[langsmith] logged scored experiment on dataset '{ds_name}'")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--transport", choices=["inprocess", "mcp"], default="inprocess")
    ap.add_argument("--category", choices=[*CATEGORIES, "all"], default="all")
    ap.add_argument("--langsmith", action="store_true", help="also log a scored LangSmith run")
    ap.add_argument("--threshold", type=float, default=0.9, help="min pass rate for exit 0")
    args = ap.parse_args()

    # Stub the live GTFS-RT feed so the suite is deterministic (and exercises the
    # fault-tolerant fallback). Override by exporting NAVIGATOR_SIMULATE_FEED=0.
    os.environ.setdefault("NAVIGATOR_SIMULATE_FEED", "1")

    if _enable_langsmith_tracing():
        print("[langsmith] tracing enabled")

    cases = cases_for(args.category)
    print(f"Running {len(cases)} cases | transport={args.transport} | "
          f"feed_simulated={os.environ.get('NAVIGATOR_SIMULATE_FEED')}")
    results = asyncio.run(_run_all(cases, args.transport))
    rate = _print_report(results)

    if args.langsmith:
        _maybe_langsmith(cases, args.transport)

    return 0 if rate >= args.threshold else 1


if __name__ == "__main__":
    raise SystemExit(main())
