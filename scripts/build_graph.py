#!/usr/bin/env python3
"""
Build the subway graph from the MTA's published GTFS static feed.

    python scripts/build_graph.py                  # download, build, write JSON
    python scripts/build_graph.py --gtfs ./gtfs    # use an already-extracted feed

Output: src/navigator/core/data/subway_graph.json — every station the MTA runs
(parent stations only), the segments between them with a per-line travel time
(so an express hop is cheaper than the local one), and the official walking
transfers. Committed to the repo so the app needs no network at runtime; re-run
this script to refresh it.

Travel times are the median observed run time on weekday trips, which is what a
rider actually experiences, rather than a hand-guessed constant.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import statistics
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

GTFS_URL = "https://rrgtfsfeeds.s3.amazonaws.com/gtfs_subway.zip"
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "src" / "navigator" / "core" / "data" / "subway_graph.json"

# The subway proper. The feed also carries the Staten Island Railway, which has
# no track connection to the rest of the system.
SIR_ROUTE = "SI"
SERVICE = "Weekday"


def _rows(folder: Path, name: str):
    with open(folder / name, newline="", encoding="utf-8-sig") as fh:
        yield from csv.DictReader(fh)


def _download(dest: Path) -> Path:
    print(f"downloading {GTFS_URL}")
    with urllib.request.urlopen(GTFS_URL, timeout=180) as resp:  # noqa: S310 - fixed MTA URL
        blob = resp.read()
    dest.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(blob)).extractall(dest)
    print(f"extracted {len(blob) / 1e6:.1f} MB to {dest}")
    return dest


def _seconds(hhmmss: str) -> int | None:
    """GTFS times run past midnight ("24:21:00"), so parse rather than strptime."""
    try:
        h, m, s = (int(part) for part in hhmmss.split(":"))
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


def build(gtfs: Path) -> dict:
    stations: dict[str, dict] = {}
    parent_of: dict[str, str] = {}
    for row in _rows(gtfs, "stops.txt"):
        if row["location_type"] == "1":
            stations[row["stop_id"]] = {
                "id": row["stop_id"], "name": row["stop_name"],
                "lat": round(float(row["stop_lat"]), 6),
                "lng": round(float(row["stop_lon"]), 6), "lines": [],
            }
        elif row["parent_station"]:
            parent_of[row["stop_id"]] = row["parent_station"]

    routes = {}
    for row in _rows(gtfs, "routes.txt"):
        routes[row["route_id"]] = {
            "name": row["route_short_name"] or row["route_id"],
            "color": "#" + (row["route_color"] or "6B7280"),
            "long_name": row["route_long_name"],
        }

    trip_route = {}
    for row in _rows(gtfs, "trips.txt"):
        if row["service_id"] == SERVICE and row["route_id"] != SIR_ROUTE:
            trip_route[row["trip_id"]] = row["route_id"]
    print(f"{len(stations)} stations · {len(routes)} routes · {len(trip_route)} weekday trips")

    # Consecutive stops of each trip become segments; collect every observed
    # run time so the median can stand in for "how long this hop takes".
    times: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    serves: dict[str, set[str]] = defaultdict(set)
    prev_trip = prev_stop = None
    prev_depart = 0
    for row in _rows(gtfs, "stop_times.txt"):
        trip = row["trip_id"]
        route = trip_route.get(trip)
        if route is None:
            prev_trip = None
            continue
        station = parent_of.get(row["stop_id"], row["stop_id"])
        if station not in stations:
            prev_trip = None
            continue
        serves[station].add(route)
        arrive = _seconds(row["arrival_time"])
        depart = _seconds(row["departure_time"])
        if trip == prev_trip and prev_stop and arrive is not None and station != prev_stop:
            minutes = (arrive - prev_depart) / 60
            if 0 < minutes <= 30:  # guard against midnight rollovers / bad rows
                times[(prev_stop, station, route)].append(minutes)
        prev_trip, prev_stop, prev_depart = trip, station, (depart if depart is not None else arrive or 0)

    edges: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for (a, b, route), samples in times.items():
        edges[(a, b)][route] = round(statistics.median(samples), 2)

    for station_id, route_ids in serves.items():
        stations[station_id]["lines"] = sorted(
            route_ids, key=lambda r: (len(r), r))  # "1" before "10", letters after digits

    # Official same-complex walking links (transfer_type 2 between *different*
    # parent stations, e.g. Lexington Av/59 St ↔ 59 St).
    transfers = []
    seen_pairs = set()
    for row in _rows(gtfs, "transfers.txt"):
        a, b = row["from_stop_id"], row["to_stop_id"]
        if a == b or a not in stations or b not in stations or (a, b) in seen_pairs:
            continue
        seen_pairs.add((a, b))
        secs = int(row.get("min_transfer_time") or 180)
        transfers.append({"from": a, "to": b, "minutes": round(max(secs, 60) / 60, 2)})

    # Stations no train serves under this feed (closed / SIR-only) are dropped.
    live = {sid for (a, b) in edges for sid in (a, b)}
    dropped = [s for s in stations if s not in live]
    for sid in dropped:
        del stations[sid]
    edges = {pair: lines for pair, lines in edges.items()
             if pair[0] in stations and pair[1] in stations}
    transfers = [t for t in transfers if t["from"] in stations and t["to"] in stations]
    print(f"dropped {len(dropped)} stations with no weekday service")

    # Station complexes: parent stations joined by in-system transfers are one
    # place to a rider (Times Sq-42 St is four parent stations). Ambiguity is
    # judged per complex, so "Times Square" resolves, while the six distinct
    # "86 St" stations across the city still ask which one.
    complex_of = _complexes(stations, transfers)
    for sid, station in stations.items():
        station["complex"] = complex_of[sid]

    # transfers.txt lists only ~150 cross-station links, far fewer than the
    # passageways that exist, which strands platforms of the same complex from
    # each other. Fill in every intra-complex pair at walking pace.
    known = {(t["from"], t["to"]) for t in transfers}
    members: dict[str, list[dict]] = defaultdict(list)
    for station in stations.values():
        members[station["complex"]].append(station)
    for group in members.values():
        for i, a in enumerate(group):
            for b in group[i + 1:]:
                if (a["id"], b["id"]) in known or (b["id"], a["id"]) in known:
                    continue
                gap = _metres(a, b)
                if gap > CONNECTED_METRES:
                    continue
                minutes = round(max(2.0, gap / WALK_METRES_PER_MIN), 2)
                transfers.append({"from": a["id"], "to": b["id"], "minutes": minutes})

    used = {r for lines in edges.values() for r in lines}
    return {
        "source": GTFS_URL,
        "service": SERVICE,
        "stations": sorted(stations.values(), key=lambda s: s["id"]),
        "edges": [{"from": a, "to": b, "lines": lines}
                  for (a, b), lines in sorted(edges.items())],
        "transfers": sorted(transfers, key=lambda t: (t["from"], t["to"])),
        "lines": {r: {**routes[r], "label": _label(r, routes[r])}
                  for r in sorted(used, key=lambda r: (len(r), r))},
    }


# Riders don't say "6X" or "GS".
_SHUTTLES = {"GS": "42 St Shuttle", "FS": "Franklin Av Shuttle", "H": "Rockaway Park Shuttle"}


def _label(route_id: str, route: dict) -> str:
    if route_id in _SHUTTLES:
        return _SHUTTLES[route_id]
    if route_id.endswith("X"):
        return f"{route_id[:-1]} express"
    return route["name"]


# Same-named parent stations this close together are platforms of one station
# (the four Fulton St halls downtown sit 22-270 m apart), not a choice a rider
# should be asked to make. Genuinely distinct namesakes — the four 125 Sts — are
# far apart and stay separate.
SAME_STATION_METRES = 400

# Brisk indoor walking pace (~4.8 km/h) for passageways between platforms.
WALK_METRES_PER_MIN = 80

# Only platforms this close are assumed to be joined by a passageway. Sharing a
# name is enough to be *one place* for a rider naming it (Penn Station), but not
# enough to imply a free transfer: 86 St on the Lexington Av line and 86 St on
# 2 Av are 350 m apart with no connection between them.
CONNECTED_METRES = 150


def _metres(a: dict, b: dict) -> float:
    """Equirectangular approximation; exact enough at this scale."""
    import math
    lat = math.radians((a["lat"] + b["lat"]) / 2)
    dx = math.radians(a["lng"] - b["lng"]) * math.cos(lat)
    dy = math.radians(a["lat"] - b["lat"])
    return math.hypot(dx, dy) * 6371000


def _complexes(stations: dict, transfers: list) -> dict[str, str]:
    """Union-find over transfer links plus same-name proximity: one id per
    walkable station complex."""
    parent = {sid: sid for sid in stations}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        a, b = find(x), find(y)
        if a != b:
            parent[a] = b

    for t in transfers:
        union(t["from"], t["to"])

    by_name: dict[str, list[dict]] = defaultdict(list)
    for station in stations.values():
        by_name[station["name"]].append(station)
    for same_name in by_name.values():
        for i, a in enumerate(same_name):
            for b in same_name[i + 1:]:
                if _metres(a, b) <= SAME_STATION_METRES:
                    union(a["id"], b["id"])

    return {sid: find(sid) for sid in stations}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtfs", type=Path, help="directory holding an extracted GTFS feed")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    gtfs = args.gtfs or _download(ROOT / ".gtfs-cache")
    graph = build(gtfs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(graph, separators=(",", ":"), sort_keys=False))
    size = args.out.stat().st_size / 1024
    print(f"wrote {args.out.relative_to(ROOT)} — {len(graph['stations'])} stations, "
          f"{len(graph['edges'])} segments, {len(graph['transfers'])} transfers, "
          f"{len(graph['lines'])} lines ({size:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
