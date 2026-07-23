"""
Dijkstra route planning over the static NYC subway graph.

Ported from the NYC Transit Hub client-side planner (src/lib/dijkstra.js) and
extended with server-side leg/transfer assignment so the agent can describe a
trip in words ("take the 1, transfer to the A at ...").

The graph is treated as undirected — exactly as the original JS relaxed both
endpoints of every edge — with the minimum weight kept when an edge is listed
in both directions.
"""
from __future__ import annotations

import heapq
from functools import lru_cache

from .graph_data import EDGES, STATION_BY_ID, STATION_IDS


# ── Adjacency, built once ──────────────────────────────────────────────
def _build_adjacency() -> dict[str, dict[str, tuple[float, int]]]:
    adj: dict[str, dict[str, tuple[float, int]]] = {sid: {} for sid in STATION_IDS}
    for a, b, minutes, stops in EDGES:
        if a not in adj or b not in adj:
            continue  # skip edges referencing unknown stations
        for u, v in ((a, b), (b, a)):
            existing = adj[u].get(v)
            if existing is None or minutes < existing[0]:
                adj[u][v] = (float(minutes), int(stops))
    return adj


_ADJ = _build_adjacency()


class RouteError(ValueError):
    """Raised when a route cannot be planned (unknown or unreachable station)."""


def _edge_lines(a: str, b: str) -> set[str]:
    """Subway lines that serve both endpoints of a hop (the ride options)."""
    return set(STATION_BY_ID[a]["lines"]) & set(STATION_BY_ID[b]["lines"])


def _dijkstra(start_id: str, end_id: str) -> tuple[list[str], float]:
    """Return (path_of_station_ids, total_minutes). Empty path if unreachable."""
    dist: dict[str, float] = {sid: float("inf") for sid in STATION_IDS}
    prev: dict[str, str | None] = {sid: None for sid in STATION_IDS}
    dist[start_id] = 0.0
    pq: list[tuple[float, str]] = [(0.0, start_id)]
    visited: set[str] = set()

    while pq:
        d, u = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)
        if u == end_id:
            break
        for v, (w, _stops) in _ADJ[u].items():
            if v in visited:
                continue
            alt = d + w
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(pq, (alt, v))

    if dist[end_id] == float("inf"):
        return [], float("inf")

    path: list[str] = []
    cur: str | None = end_id
    while cur is not None:
        path.insert(0, cur)
        cur = prev[cur]
    return path, dist[end_id]


def _assign_legs(path: list[str]) -> list[dict]:
    """
    Split a station path into ride legs, minimizing line changes.

    Greedy interval-intersection: extend a leg while a single line can serve
    every hop so far; when the feasible set empties, that boundary is a transfer.
    """
    legs: list[dict] = []
    n = len(path)
    idx = 0
    while idx < n - 1:
        feasible = _edge_lines(path[idx], path[idx + 1])
        j = idx + 1
        cur = set(feasible)
        while j < n - 1:
            nxt = _edge_lines(path[j], path[j + 1])
            inter = cur & nxt
            if not inter:
                break
            cur = inter
            j += 1

        minutes, stops = 0.0, 0
        for k in range(idx, j):
            w, s = _ADJ[path[k]][path[k + 1]]
            minutes += w
            stops += s

        legs.append({
            "line": sorted(cur)[0] if cur else "?",
            "line_options": sorted(cur),
            "from_id": path[idx],
            "from": STATION_BY_ID[path[idx]]["name"],
            "to_id": path[j],
            "to": STATION_BY_ID[path[j]]["name"],
            "num_stops": stops,
            "minutes": round(minutes, 1),
        })
        idx = j
    return legs


def _summary(legs: list[dict], minutes: float, stops: int) -> str:
    if not legs:
        return "No route found."
    parts = []
    for i, leg in enumerate(legs):
        verb = "Take" if i == 0 else f"transfer to"
        parts.append(
            f"{verb} the {leg['line']} from {leg['from']} to {leg['to']} "
            f"({leg['num_stops']} stop{'s' if leg['num_stops'] != 1 else ''})"
        )
    transfers = len(legs) - 1
    tx = "no transfers" if transfers == 0 else f"{transfers} transfer{'s' if transfers != 1 else ''}"
    return (
        "; ".join(parts)
        + f". About {int(round(minutes))} min, {stops} stops, {tx}."
    )


@lru_cache(maxsize=512)
def plan_route(start_id: str, end_id: str) -> dict:
    """
    Plan the fastest subway route between two station ids.

    Returns a dict with the path, per-leg directions, total time, stop count,
    and transfer count. Raises RouteError for unknown/unreachable stations.
    """
    if start_id not in STATION_BY_ID:
        raise RouteError(f"unknown start station id: {start_id!r}")
    if end_id not in STATION_BY_ID:
        raise RouteError(f"unknown end station id: {end_id!r}")
    if start_id == end_id:
        return {
            "found": True,
            "path": [start_id],
            "stations": [STATION_BY_ID[start_id]["name"]],
            "total_minutes": 0,
            "total_stops": 0,
            "num_transfers": 0,
            "legs": [],
            "summary": "You are already there.",
        }

    path, minutes = _dijkstra(start_id, end_id)
    if not path:
        raise RouteError(
            f"no route between {STATION_BY_ID[start_id]['name']} and "
            f"{STATION_BY_ID[end_id]['name']} in the modeled network"
        )

    legs = _assign_legs(path)
    total_stops = sum(leg["num_stops"] for leg in legs)
    return {
        "found": True,
        "path": path,
        "stations": [STATION_BY_ID[sid]["name"] for sid in path],
        "total_minutes": int(round(minutes)),
        "total_stops": total_stops,
        "num_transfers": max(0, len(legs) - 1),
        "legs": legs,
        "summary": _summary(legs, minutes, total_stops),
    }
