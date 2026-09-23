"""
Offline geocoding over the modeled network.

Two jobs the routing layer needs but the raw graph doesn't provide:

  1. resolve_station(text)  — fuzzy free-text → a station id in the graph
  2. geocode_place(text)    — free-text place/landmark → (lat, lng) + nearest station

Both are deterministic and dependency-free (difflib + a small landmark table),
so tool-calls are reproducible in tests and evals without a network round-trip.
A real reverse-geocoder could be dropped in behind the same interface later.
"""
from __future__ import annotations

import difflib
import math
import re

from .graph_data import COMPLEX_MEMBERS, STATION_BY_ID, STATIONS, connected_members, feed_line

# A handful of well-known NYC landmarks → coordinates. Deliberately small and
# offline; each resolves to its nearest modeled station via haversine.
LANDMARKS: dict[str, tuple[float, float]] = {
    "times square": (40.7580, -73.9855),
    "grand central": (40.7527, -73.9772),
    "penn station": (40.7506, -73.9935),
    "empire state building": (40.7484, -73.9857),
    "bryant park": (40.7536, -73.9832),
    "union square": (40.7359, -73.9906),
    "world trade center": (40.7127, -74.0099),
    "wall street": (40.7074, -74.0114),
    "brooklyn bridge": (40.7061, -73.9969),
    "central park": (40.7681, -73.9819),
    "columbia university": (40.8075, -73.9626),
    "coney island": (40.5755, -73.9707),
    "barclays center": (40.6826, -73.9754),
    "yankee stadium": (40.8296, -73.9262),
    "jfk airport": (40.6413, -73.7781),
    "laguardia airport": (40.7769, -73.8740),
    "flushing": (40.7596, -73.8300),
    "jackson heights": (40.7466, -73.8912),
    "prospect park": (40.6602, -73.9690),
    "williamsburg": (40.7081, -73.9571),
    "harlem": (40.8116, -73.9465),
    "greenwich village": (40.7336, -74.0027),
    "soho": (40.7233, -74.0030),
    "chinatown": (40.7158, -73.9970),
    "midtown": (40.7549, -73.9840),
    "lincoln center": (40.7725, -73.9835),
    "rockefeller center": (40.7587, -73.9787),
}


# A landmark is only snapped to a station within walking distance; beyond this
# (e.g. JFK, 8 km from the nearest modeled stop) a route would be misleading.
MAX_SNAP_KM = 2.0


class GeocodeError(ValueError):
    """Raised when a place/station string cannot be resolved to the graph."""


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometers."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


# Canonical spellings, so "Union Square" and "14 St-Union Sq" meet in the
# middle. These are folded rather than deleted: dropping "square" outright made
# "Union Square" match "Union St".
_SYNONYMS = {
    "street": "st", "avenue": "av", "ave": "av", "square": "sq", "road": "rd",
    "place": "pl", "boulevard": "blvd", "parkway": "pkwy", "center": "ctr",
    "centre": "ctr", "heights": "hts", "junction": "jct", "terminal": "term",
    "island": "is", "beach": "bch", "fort": "ft", "mount": "mt", "saint": "st",
}
_NOISE = {"the", "station", "stop", "subway"}


def _tokens(text: str) -> list[str]:
    text = text.lower().strip().replace("–", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    out = []
    for word in text.split():
        word = _SYNONYMS.get(word, word)
        if word not in _NOISE:
            out.append(word)
    return out


def _normalize(text: str) -> str:
    return " ".join(_tokens(text))


_ACRONYMS = {"jfk"}


def display_name(name: str) -> str:
    """Title-case a landmark key for answers ("jfk airport" → "JFK Airport")."""
    return " ".join(w.upper() if w in _ACRONYMS else w.capitalize() for w in name.split())


def normalize(text: str) -> str:
    """Public alias: the canonical form used for all fuzzy place matching."""
    return _normalize(text)


def nearest_station(lat: float, lng: float) -> dict:
    """Nearest modeled station to a coordinate, with distance_km added."""
    best = min(
        STATIONS,
        key=lambda s: haversine_km(lat, lng, s["lat"], s["lng"]),
    )
    out = dict(best)
    out["distance_km"] = round(haversine_km(lat, lng, best["lat"], best["lng"]), 3)
    return out


def _score(query_tokens: list[str], name_tokens: list[str]) -> float:
    """How well a query matches a station name, 0-1.

    Whole-token containment, not raw substring: "penn" must not match
    "Pennsylvania Av" just because the letters line up.
    """
    if query_tokens == name_tokens:
        return 1.0
    if name_tokens[:len(query_tokens)] == query_tokens:
        return 0.95  # "Grand Central" ⊂ "Grand Central-42 St"
    if set(query_tokens) <= set(name_tokens):
        return 0.9   # "Union Sq" ⊂ "14 St-Union Sq"
    return difflib.SequenceMatcher(None, " ".join(query_tokens), " ".join(name_tokens)).ratio()


def _pick(station_ids: list[str], query_tokens: list[str]) -> dict:
    """The station a rider means within one complex: whichever name matches the
    query best, then the one serving the most lines. Without the name check,
    "World Trade Center" answers with its complex-mate Cortlandt St."""
    return max((STATION_BY_ID[sid] for sid in station_ids),
               key=lambda s: (_score(query_tokens, _tokens(s["name"])), len(s["lines"]), s["id"]))


def resolve_station(query: str) -> dict:
    """
    Resolve free text to a single best-matching station.

    Ambiguity is judged per station *complex*: the four parent stations that
    make up Times Sq-42 St are one place and resolve cleanly, while the six
    unrelated "86 St" stations across the city come back as candidates to pick
    between.

    Returns {"station": {...}, "candidates": [ids...], "ambiguous": bool}.
    Raises GeocodeError if nothing plausibly matches.
    """
    q = _tokens(query)
    if not q:
        raise GeocodeError("empty station query")

    scored = sorted(((_score(q, _tokens(s["name"])), s) for s in STATIONS),
                    key=lambda t: t[0], reverse=True)
    best_ratio, best = scored[0]
    # Real station queries score >=0.9 via the containment rules above. The 0.6
    # floor plus a shared-token requirement rejects near-homophone junk:
    # "Atlantis" scores 0.74 against "Atlantic Av" but shares no word with it.
    if best_ratio < 0.6 or not (set(q) & set(_tokens(best["name"]))):
        raise GeocodeError(f"no station matches {query!r}")

    # A containment-grade match (>=0.9) only competes with other containment
    # matches, so "Union Square" isn't dragged into a choice with "Union St".
    cutoff = max(0.9, best_ratio - 0.05) if best_ratio >= 0.9 else best_ratio - 0.05
    by_complex: dict[str, float] = {}
    for ratio, station in scored:
        if ratio < cutoff:
            break
        by_complex.setdefault(station["complex"], ratio)
    candidates = [_pick(COMPLEX_MEMBERS[cid], q)["id"] for cid in list(by_complex)[:4]]
    return {
        "station": STATION_BY_ID[candidates[0]],
        "candidates": candidates,
        "ambiguous": len(candidates) > 1,
        "score": round(best_ratio, 3),
    }


def complex_lines(station_id: str) -> list[str]:
    """Every line reachable on foot from this station without leaving the system.

    A rider asking what stops at Union Sq means all of 4/5/6/L/N/Q/R/W, not one
    platform's lines — but not the unconnected 86 St across town either."""
    members = connected_members(station_id)
    lines: list[str] = []
    for sid in members:
        for line in STATION_BY_ID[sid]["lines"]:
            if line not in lines:
                lines.append(line)
    return sorted(lines, key=lambda line: (len(line), line))


def rider_lines(lines) -> list[str]:
    """Lines as a rider names them: the schedule's express variants (6X, FX) and
    the three shuttles collapse onto the 6, the F and the S."""
    out: list[str] = []
    for line in lines or []:
        name = feed_line(line)
        if name not in out:
            out.append(name)
    return out


def _candidate(station_id: str) -> dict:
    """A choice offered to the rider. Same-named stations are told apart by the
    lines they serve, the way New Yorkers do it ("86 St on the 4/5/6")."""
    s = STATION_BY_ID[station_id]
    return {"id": s["id"], "name": s["name"], "lines": rider_lines(complex_lines(station_id))}


def resolve_endpoint(query: str) -> dict:
    """
    Resolve a trip endpoint: a station name, else a landmark or neighborhood
    mapped to its nearest station ("Williamsburg" → Marcy Av).

    Returns {"id", "name", "ambiguous", "candidates": [...], "via"} where `via`
    is None for a direct station match, or {"place", "distance_km"} when a
    landmark was snapped to a station — so the answer can say so rather than
    silently substituting. Raises GeocodeError if neither resolves, including
    when the nearest station is too far to be a useful answer.
    """
    query = str(query).strip()
    if query in STATION_BY_ID:  # already a station id (models pass ids back)
        s = STATION_BY_ID[query]
        return {"id": s["id"], "name": s["name"], "ambiguous": False,
                "candidates": [], "via": None}

    station = None
    try:
        station = resolve_station(query)
    except GeocodeError:
        pass

    # A merely fuzzy station match loses to a real landmark: "Central Park" is
    # the park, not the Central Av stop in Bushwick. A containment-grade match
    # (>=0.9) is taken as the rider naming the station itself.
    if station is None or station["score"] < 0.9:
        place = None
        try:
            place = geocode_place(query)
        except GeocodeError:
            pass
        if place is not None and (place["source"] == "landmark" or station is None):
            near = place["nearest_station"]
            if near["distance_km"] > MAX_SNAP_KM:
                raise GeocodeError(
                    f"{display_name(place['matched'])} is {near['distance_km']:.1f} km from the "
                    f"nearest station in the modeled network ({near['name']}), too far to plan "
                    f"a reliable subway trip")
            return {"id": near["id"], "name": near["name"], "ambiguous": False,
                    "candidates": [],
                    "via": {"place": place["matched"], "distance_km": near["distance_km"]}}

    if station is None:
        raise GeocodeError(f"no station or place matches {query!r}")
    return {
        "id": station["station"]["id"],
        "name": station["station"]["name"],
        "ambiguous": station["ambiguous"],
        "candidates": [_candidate(sid) for sid in station["candidates"]],
        "via": None,
    }


def geocode_place(query: str) -> dict:
    """
    Resolve a free-text place to coordinates and the nearest station.

    Tries the landmark table first, then falls back to station-name resolution.
    """
    q = _normalize(query)
    # Landmark match (exact-normalized, then fuzzy over landmark keys)
    for name, (lat, lng) in LANDMARKS.items():
        if _normalize(name) == q:
            near = nearest_station(lat, lng)
            return {"query": query, "lat": lat, "lng": lng, "source": "landmark",
                    "matched": name, "nearest_station": near}

    landmark_keys = list(LANDMARKS)
    match = difflib.get_close_matches(q, [_normalize(k) for k in landmark_keys], n=1, cutoff=0.7)
    if match:
        idx = [_normalize(k) for k in landmark_keys].index(match[0])
        name = landmark_keys[idx]
        lat, lng = LANDMARKS[name]
        near = nearest_station(lat, lng)
        return {"query": query, "lat": lat, "lng": lng, "source": "landmark",
                "matched": name, "nearest_station": near}

    # Fall back to a station name
    resolved = resolve_station(query)
    s = resolved["station"]
    return {"query": query, "lat": s["lat"], "lng": s["lng"], "source": "station",
            "matched": s["name"], "nearest_station": {**s, "distance_km": 0.0}}
