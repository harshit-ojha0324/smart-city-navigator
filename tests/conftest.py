import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))


def station_id(name: str) -> str:
    """Resolve a station name to its GTFS id, so tests read by name rather than
    pinning opaque ids like "R16"."""
    from navigator.core import geocode

    return geocode.resolve_station(name)["station"]["id"]


# Hermetic tests: never touch the live MTA network — use the deterministic
# simulation fallback (also proves the fault-tolerant degraded path).
os.environ.setdefault("NAVIGATOR_SIMULATE_FEED", "1")
