import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

# Hermetic tests: never touch the live MTA network — use the deterministic
# simulation fallback (also proves the fault-tolerant degraded path).
os.environ.setdefault("NAVIGATOR_SIMULATE_FEED", "1")
