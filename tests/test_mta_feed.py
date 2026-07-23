from navigator.core import mta_feed
from navigator.core.cache import TTLCache


def test_simulated_alerts_cover_all_lines():
    alerts = mta_feed._simulated_alerts()
    assert set(alerts) == set(mta_feed.SUBWAY_LINE_IDS)
    for info in alerts.values():
        assert 0 <= info["severity"] <= 3


def test_simulation_is_deterministic_within_a_minute():
    a = mta_feed._simulated_alerts()
    b = mta_feed._simulated_alerts()
    # Same minute seed → identical severities.
    assert {k: v["severity"] for k, v in a.items()} == {k: v["severity"] for k, v in b.items()}


def test_get_line_status_valid_and_invalid():
    ok = mta_feed.get_line_status("A")
    assert ok["line"] == "A" and "severity" in ok
    bad = mta_feed.get_line_status("QZ")
    assert "error" in bad and "QZ" in bad["error"]


def test_status_summary_is_text():
    s = mta_feed.status_summary()
    assert isinstance(s, str) and len(s) > 0


def test_ttl_cache_memory_backend():
    c = TTLCache(redis_url=None)
    assert c.backend == "memory"
    c.set("k", {"v": 1}, ttl=60)
    assert c.get("k") == {"v": 1}
    c.delete("k")
    assert c.get("k") is None


def test_ttl_cache_expiry():
    c = TTLCache(redis_url=None)
    c.set("k", 42, ttl=0)  # ttl=0 → no expiry sentinel; still retrievable
    assert c.get("k") == 42
    c._mem["k2"] = (99, 1.0)  # already-expired timestamp
    assert c.get("k2") is None
