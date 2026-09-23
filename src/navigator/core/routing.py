"""
Transfer-aware route planning over the GTFS subway network.

Dijkstra runs over (station, line-you-are-riding) states rather than plain
stations, which buys three things a station-only search can't express:

  * a change of train costs TRANSFER_MINUTES, so the planner never trades a
    one-seat ride for two changes to save a minute;
  * each segment is priced on the line actually ridden, so an express hop beats
    the local one over the same pair of stations;
  * in-system walking transfers (Times Sq ↔ Port Authority) are first-class
    moves, and come back as explicit walking legs.

Legs fall straight out of the optimal path, so the agent can describe the trip
in words ("take the L, transfer to the N at Union Sq").
"""
from __future__ import annotations

import heapq
from functools import lru_cache

from .graph_data import (
    RIDE_EDGES,
    STATION_BY_ID,
    STATION_IDS,
    WALK_EDGES,
    line_label,
    same_complex,
)

# Minutes charged for changing trains: the platform change plus the average
# wait for the next one.
TRANSFER_MINUTES = 4.0

# Pseudo-line for a walking connection inside a station complex.
WALK = "walk"


# ── Adjacency, built once ──────────────────────────────────────────────
def _build_adjacency() -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, dict[str, float]]]:
    rides: dict[str, dict[str, dict[str, float]]] = {sid: {} for sid in STATION_IDS}
    walks: dict[str, dict[str, float]] = {sid: {} for sid in STATION_IDS}
    for edge in RIDE_EDGES:
        a, b = edge["from"], edge["to"]
        if a in rides and b in rides:
            rides[a][b] = {line: float(m) for line, m in edge["lines"].items()}
    for edge in WALK_EDGES:
        a, b, minutes = edge["from"], edge["to"], float(edge["minutes"])
        if a in walks and b in walks:
            walks[a][b] = min(minutes, walks[a].get(b, minutes))
            walks[b][a] = min(minutes, walks[b].get(a, minutes))
    return rides, walks


_RIDES, _WALKS = _build_adjacency()


class RouteError(ValueError):
    """Raised when a route cannot be planned (unknown or unreachable station)."""


def _dijkstra(start_id: str, end_id: str) -> tuple[list[tuple[str, str, str]], float]:
    """
    Shortest path over (station, line) states.

    Returns ([(station_id, how_you_arrived, line), ...], total_minutes) where
    `how_you_arrived` is "ride", "walk", or "start"; empty path if unreachable.
    Ties break toward fewer transfers.
    """
    start = (start_id, None)
    best: dict[tuple, tuple[float, int]] = {start: (0.0, 0)}
    prev: dict[tuple, tuple[tuple, str, str]] = {}
    seq = 0  # heap tie-breaker so states are never compared
    pq: list = [(0.0, 0, seq, start)]
    done: set[tuple] = set()

    while pq:
        cost, transfers, _, state = heapq.heappop(pq)
        if state in done:
            continue
        done.add(state)
        station, riding = state

        if station == end_id:
            # Each entry carries the move that *arrived* at it, so consecutive
            # pairs read as (where you were, how you got to the next station).
            path: list[tuple[str, str, str]] = []
            cur = state
            while cur in prev:
                pred, how, line = prev[cur]
                path.append((cur[0], how, line))
                cur = pred
            path.append((cur[0], "start", None))
            path.reverse()
            return path, cost

        for nxt, lines in _RIDES[station].items():
            for line, minutes in lines.items():
                change = riding is not None and line != riding
                cand = (cost + minutes + (TRANSFER_MINUTES if change else 0.0),
                        transfers + int(change))
                ns = (nxt, line)
                if ns not in done and cand < best.get(ns, (float("inf"), 0)):
                    best[ns] = cand
                    prev[ns] = (state, "ride", line)
                    seq += 1
                    heapq.heappush(pq, (cand[0], cand[1], seq, ns))

        # Walking keeps whatever line you were riding: stepping across a complex
        # and re-boarding the same line is not a transfer.
        for nxt, minutes in _WALKS[station].items():
            cand = (cost + minutes, transfers)
            ns = (nxt, riding)
            if ns not in done and cand < best.get(ns, (float("inf"), 0)):
                best[ns] = cand
                prev[ns] = (state, "walk", WALK)
                seq += 1
                heapq.heappush(pq, (cand[0], cand[1], seq, ns))

    return [], float("inf")


def _legs_from_path(path: list[tuple[str, str, str]]) -> list[dict]:
    """Group the path's moves into ride legs (consecutive hops on one line)
    and walking legs."""
    legs: list[dict] = []
    carried = 0.0     # in-station transfer time, folded into the leg it serves
    boarding = None   # platform you walked to inside the complex
    for (a, _, _), (b, how, line) in zip(path, path[1:], strict=False):  # pairwise
        walk = how == "walk"
        minutes = (_WALKS[a][b] if walk else _RIDES[a][b][line])
        if walk and same_complex(a, b):
            # Crossing platforms inside one station is the transfer itself, not
            # a walking leg ("walk from Times Sq-42 St to Times Sq-42 St").
            carried += minutes
            boarding = b
            continue
        if boarding is not None:
            # Both sides of an in-station change name the same platform, so the
            # trip reads "...to 14 St; transfer to the F from 14 St".
            if legs:
                legs[-1]["to_id"] = boarding
                legs[-1]["to"] = STATION_BY_ID[boarding]["name"]
            a = boarding
            boarding = None
        if legs and legs[-1]["line"] == line and not (walk and legs[-1]["walk"]):
            leg = legs[-1]
        else:
            leg = {"line": line, "label": "walk" if walk else line_label(line),
                   "walk": walk, "from_id": a, "from": STATION_BY_ID[a]["name"],
                   "num_stops": 0, "minutes": 0.0}
            legs.append(leg)
        leg["to_id"], leg["to"] = b, STATION_BY_ID[b]["name"]
        leg["minutes"] = round(leg["minutes"] + minutes + carried, 1)
        carried = 0.0
        if not walk:
            leg["num_stops"] += 1
    if carried and legs:  # trailing platform change at the destination
        legs[-1]["minutes"] = round(legs[-1]["minutes"] + carried, 1)
        if boarding is not None:
            legs[-1]["to_id"] = boarding
            legs[-1]["to"] = STATION_BY_ID[boarding]["name"]
    return legs


def _summary(legs: list[dict], minutes: float, stops: int) -> str:
    if not legs:
        return "No route found."
    parts = []
    for i, leg in enumerate(legs):
        if leg["walk"]:
            parts.append(f"walk from {leg['from']} to {leg['to']} (~{int(round(leg['minutes']))} min)")
            continue
        verb = "Take" if i == 0 else ("take" if legs[i - 1]["walk"] else "transfer to")
        parts.append(
            f"{verb} the {leg['label']} from {leg['from']} to {leg['to']} "
            f"({leg['num_stops']} stop{'s' if leg['num_stops'] != 1 else ''})"
        )
    transfers = sum(1 for leg in legs[1:] if not leg["walk"])
    tx = "no transfers" if transfers == 0 else f"{transfers} transfer{'s' if transfers != 1 else ''}"
    text = "; ".join(parts)
    return (
        text[0].upper() + text[1:]
        + f". About {int(round(minutes))} min, {stops} stop{'s' if stops != 1 else ''}, {tx}."
    )


@lru_cache(maxsize=512)
def plan_route(start_id: str, end_id: str) -> dict:
    """
    Plan the fastest subway route between two station ids.

    Returns a dict with the path, per-leg directions, total time (ride time plus
    TRANSFER_MINUTES per change), stop count, and transfer count. Raises
    RouteError for unknown/unreachable stations.
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
            "transfer_minutes": TRANSFER_MINUTES,
            "legs": [],
            "summary": "You are already there.",
        }

    states, minutes = _dijkstra(start_id, end_id)
    if not states:
        raise RouteError(
            f"no route between {STATION_BY_ID[start_id]['name']} and "
            f"{STATION_BY_ID[end_id]['name']} in the modeled network"
        )

    path = [sid for sid, _how, _line in states]
    legs = _legs_from_path(states)
    total_stops = sum(leg["num_stops"] for leg in legs)
    return {
        "found": True,
        "path": path,
        "stations": [STATION_BY_ID[sid]["name"] for sid in path],
        "total_minutes": int(round(minutes)),
        "total_stops": total_stops,
        "num_transfers": sum(1 for leg in legs[1:] if not leg["walk"]),
        "transfer_minutes": TRANSFER_MINUTES,
        "legs": legs,
        "summary": _summary(legs, minutes, total_stops),
    }
