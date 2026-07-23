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

from .graph_data import STATIONS, STATION_BY_ID


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


def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = text.replace("–", " ").replace("-", " ")
    text = re.sub(r"\b(st|street|av|ave|avenue|sq|square|station|the)\b", " ", text)
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def nearest_station(lat: float, lng: float) -> dict:
    """Nearest modeled station to a coordinate, with distance_km added."""
    best = min(
        STATIONS,
        key=lambda s: haversine_km(lat, lng, s["lat"], s["lng"]),
    )
    out = dict(best)
    out["distance_km"] = round(haversine_km(lat, lng, best["lat"], best["lng"]), 3)
    return out


def resolve_station(query: str) -> dict:
    """
    Resolve free text to a single best-matching station.

    Returns {"station": {...}, "candidates": [ids...], "ambiguous": bool}.
    Raises GeocodeError if nothing plausibly matches.
    """
    q = _normalize(query)
    if not q:
        raise GeocodeError("empty station query")

    scored: list[tuple[float, dict]] = []
    for s in STATIONS:
        name_norm = _normalize(s["name"])
        ratio = difflib.SequenceMatcher(None, q, name_norm).ratio()
        # Boost substring / token containment (e.g. "times sq" ⊂ "times sq 42 st")
        if q in name_norm or name_norm.startswith(q):
            ratio = max(ratio, 0.9)
        if all(tok in name_norm.split() for tok in q.split()):
            ratio = max(ratio, 0.8)
        scored.append((ratio, s))

    scored.sort(key=lambda t: t[0], reverse=True)
    best_ratio, best = scored[0]
    # Real station queries score >=0.8 via the substring/token boosts above; the
    # 0.6 floor rejects near-homophone junk ("Atlantis" -> "Atlantic Av").
    if best_ratio < 0.6:
        raise GeocodeError(f"no station matches {query!r}")

    close = [s["id"] for r, s in scored if r >= best_ratio - 0.05][:4]
    return {
        "station": best,
        "candidates": close,
        "ambiguous": len(close) > 1,
        "score": round(best_ratio, 3),
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
