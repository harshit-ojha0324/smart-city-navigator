from navigator.core import mta_feed


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
    s = mta_feed.status_summary(mta_feed.fetch_alerts())
    assert isinstance(s, str) and len(s) > 0


def test_alerts_are_cached_until_the_ttl_expires(monkeypatch):
    """The cache is one tuple now; it must still hold for ALERTS_TTL and then refetch."""
    calls = []
    monkeypatch.setattr(mta_feed, "_simulated_alerts",
                        lambda: calls.append(1) or {"L": {"severity": 0}})
    monkeypatch.setattr(mta_feed, "_cached", None)
    monkeypatch.setenv("NAVIGATOR_SIMULATE_FEED", "1")

    mta_feed.fetch_alerts()
    mta_feed.fetch_alerts()
    assert len(calls) == 1, "second call inside the TTL must come from cache"

    monkeypatch.setattr(mta_feed, "_cached", (0.0, {"stale": True}))  # expired
    mta_feed.fetch_alerts()
    assert len(calls) == 2, "an expired entry must be refetched"
