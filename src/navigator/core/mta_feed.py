"""
Live MTA GTFS-RT feed access.

Ported from the NYC Transit Hub backend (backend/services/mta_feed.py):
  - Service alerts             via JSON feed (camsys/subway-alerts.json)
  - Elevator/escalator outages via JSON feed (nyct/nyct_ene.json)

No API key is required — these MTA feeds are public. Every fetch is wrapped in a
TTL cache and falls back to a deterministic, minute-seeded simulation on any
network/parse error, so the alerts MCP server keeps answering through an
upstream outage (the fault-tolerance the resume claims, exercised by tests).
"""
from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone

import requests

from .cache import TTLCache

# ── Feed URLs ──────────────────────────────────────────────────────────
_BASE = "https://api-endpoint.mta.info/Dataservice/mtagtfsfeeds"
SUBWAY_ALERTS_URL = f"{_BASE}/camsys%2Fsubway-alerts.json"
ELEVATOR_OUTAGES_URL = f"{_BASE}/nyct%2Fnyct_ene.json"

SUBWAY_LINE_IDS = [
    "1", "2", "3", "4", "5", "6", "7",
    "A", "C", "E", "B", "D", "F", "M",
    "G", "J", "Z", "L", "N", "Q", "R", "W", "S",
]

# MTA Mercury alert_type → (severity 0-3, human label)
ALERT_TYPE_MAP = {
    "Planned - Suspended": (3, "Suspended"),
    "No Scheduled Service": (3, "Suspended"),
    "Planned - Part Suspended": (2, "Service Change"),
    "Planned - Reroute": (2, "Service Change"),
    "Planned - Stops Skipped": (2, "Planned Work"),
    "Planned - Express to Local": (2, "Planned Work"),
    "Reduced Service": (1, "Delays"),
    "Special Schedule": (1, "Delays"),
    "Boarding Change": (1, "Service Change"),
    "Extra Service": (0, "Good Service"),
    "Station Notice": (0, "Good Service"),
}

STATUS_MESSAGES = {
    0: "Trains are running on schedule.",
    1: "Expect delays. Check MTA.info for updates.",
    2: "Service changes in effect. Check MTA.info for details.",
    3: "Service suspended between select stations. Shuttle buses in operation.",
}

# Set NAVIGATOR_SIMULATE_FEED=1 to force the offline simulation path (used by the
# tool-failure eval scenarios and by CI, which has no outbound network).
_SIMULATE_ENV = "NAVIGATOR_SIMULATE_FEED"

_cache = TTLCache(os.environ.get("REDIS_URL"))
ALERTS_TTL = int(os.environ.get("ALERTS_TTL", "30"))


def _now_str() -> str:
    return datetime.now(timezone.utc).strftime("%I:%M:%S %p")


def _good_status() -> dict:
    return {"severity": 0, "status": "Good Service",
            "message": STATUS_MESSAGES[0], "updatedAt": _now_str()}


def _mta_get(url: str, timeout: int = 10):
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _get_translation(text_obj: dict) -> str:
    translations = (text_obj or {}).get("translation", [])
    for t in translations:
        if t.get("language", "").startswith("en"):
            return t.get("text", "")
    return translations[0].get("text", "") if translations else ""


def _simulate() -> bool:
    return os.environ.get(_SIMULATE_ENV, "").strip() in {"1", "true", "yes"}


# ── Service alerts ─────────────────────────────────────────────────────
def fetch_alerts(*, use_cache: bool = True) -> dict:
    """Return {line_id: {severity, status, message, updatedAt}} for all lines."""
    if use_cache:
        cached = _cache.get("alerts")
        if cached is not None:
            return cached

    if _simulate():
        result = _simulated_alerts()
    else:
        try:
            result = _parse_alerts_json(_mta_get(SUBWAY_ALERTS_URL))
        except Exception as exc:  # network/parse — degrade, never crash
            print(f"[mta_feed] alerts fetch error: {exc} — using simulation")
            result = _simulated_alerts()

    if use_cache:
        _cache.set("alerts", result, ttl=ALERTS_TTL)
    return result


def get_line_status(line_id: str) -> dict:
    """Status for a single line id (case-insensitive)."""
    alerts = fetch_alerts()
    line_id = line_id.strip().upper()
    if line_id not in alerts:
        return {"line": line_id, "error": f"unknown subway line {line_id!r}",
                "known_lines": SUBWAY_LINE_IDS}
    return {"line": line_id, **alerts[line_id]}


def _parse_alerts_json(data: dict) -> dict:
    now = _now_str()
    result = {line: _good_status() for line in SUBWAY_LINE_IDS}
    for entity in data.get("entity", []):
        alert = entity.get("alert", {})
        mercury = alert.get("transit_realtime.mercury_alert", {})
        severity, label = ALERT_TYPE_MAP.get(mercury.get("alert_type", ""), (0, "Good Service"))
        header = _get_translation(alert.get("header_text", {}))
        description = _get_translation(alert.get("description_text", {}))
        message = header or description or STATUS_MESSAGES[severity]
        for informed in alert.get("informed_entity", []):
            route_id = informed.get("route_id", "")
            if route_id not in result:
                continue
            if severity > result[route_id]["severity"]:  # only upgrade severity
                result[route_id] = {"severity": severity, "status": label,
                                    "message": message, "updatedAt": now}
    return result


# ── Elevator / escalator outages ───────────────────────────────────────
def fetch_elevator_outages() -> list:
    if _simulate():
        return []
    try:
        return _parse_elevator_json(_mta_get(ELEVATOR_OUTAGES_URL))
    except Exception as exc:
        print(f"[mta_feed] elevator fetch error: {exc}")
        return []


def _parse_elevator_json(data: list) -> list:
    outages = []
    for item in data:
        eq_type = item.get("equipmenttype", "").upper()
        outages.append({
            "equipment": item.get("equipment", ""),
            "type": "Elevator" if eq_type == "EL" else "Escalator",
            "station": item.get("station", ""),
            "lines": item.get("linesservedbyelevator", ""),
            "borough": item.get("borough", ""),
            "serving": item.get("serving", ""),
            "reason": item.get("reason", ""),
            "eta": item.get("estimatedreturntoservice", ""),
            "ada": item.get("ADA", "N") == "Y",
        })
    return outages


# ── Deterministic simulation fallback ──────────────────────────────────
def _simulated_alerts() -> dict:
    """Minute-seeded, deterministic status for every line (offline fallback)."""
    now = datetime.now(timezone.utc)
    minute_seed = int(now.timestamp() // 60)
    updated_at = now.strftime("%I:%M:%S %p")
    result = {}
    for line in SUBWAY_LINE_IDS:
        seed_val = int(hashlib.md5(f"{line}{minute_seed}".encode()).hexdigest(), 16)
        bucket = seed_val % 100
        if bucket < 60:
            sev, label = 0, "Good Service"
        elif bucket < 78:
            sev, label = 1, "Delays"
        elif bucket < 90:
            sev, label = 2, "Planned Work"
        elif bucket < 96:
            sev, label = 2, "Service Change"
        else:
            sev, label = 3, "Suspended"
        result[line] = {"severity": sev, "status": label,
                        "message": STATUS_MESSAGES[sev], "updatedAt": updated_at}
    return result


def status_summary(alerts: dict | None = None) -> str:
    """Compact human summary of current status — used for LLM context injection."""
    alerts = alerts or fetch_alerts()
    good, affected = [], []
    for line_id, info in sorted(alerts.items()):
        if info.get("severity", 0) == 0:
            good.append(line_id)
        else:
            affected.append(
                f"  {line_id}: {info.get('status')} "
                f"(severity {info.get('severity')}/3) — {info.get('message')}"
            )
    parts = []
    if affected:
        parts.append("Lines with active issues:\n" + "\n".join(affected))
    if good:
        parts.append("Lines running normally: " + ", ".join(good))
    return "\n\n".join(parts) if parts else "All lines reporting Good Service."
