import pytest

from navigator.core import geocode


def test_resolve_exact_station():
    res = geocode.resolve_station("Times Square")
    assert res["station"]["id"] == "ts"
    assert not res["ambiguous"]


def test_resolve_ambiguous_station():
    res = geocode.resolve_station("125 St")
    assert res["ambiguous"]
    assert len(res["candidates"]) >= 2
    assert "har" in res["candidates"] or "125_123" in res["candidates"]


def test_resolve_rejects_junk():
    with pytest.raises(geocode.GeocodeError):
        geocode.resolve_station("Atlantis")
    with pytest.raises(geocode.GeocodeError):
        geocode.resolve_station("Narnia")


def test_nearest_station_to_coordinate():
    near = geocode.nearest_station(40.7496, -73.9879)  # Herald Sq coords
    assert near["id"] == "hz"
    assert near["distance_km"] < 0.2


def test_geocode_landmark_to_nearest_station():
    g = geocode.geocode_place("Empire State Building")
    assert g["source"] == "landmark"
    assert g["nearest_station"]["id"] == "hz"


def test_geocode_fuzzy_landmark():
    g = geocode.geocode_place("empire state")
    assert g["matched"] == "empire state building"


def test_haversine_known_values():
    assert geocode.haversine_km(40.0, -73.0, 40.0, -73.0) == 0.0
    # ~1 degree of latitude ≈ 111 km
    assert 110 < geocode.haversine_km(40.0, -73.0, 41.0, -73.0) < 112
