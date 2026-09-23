import pytest

from navigator.core import geocode


def test_resolve_exact_station():
    res = geocode.resolve_station("Times Square")
    assert res["station"]["name"] == "Times Sq-42 St"
    assert not res["ambiguous"]


def test_one_complex_is_one_place():
    # Times Sq is four GTFS parent stations plus Port Authority; a rider naming
    # it should not be asked which platform they meant.
    res = geocode.resolve_station("Times Square")
    assert not res["ambiguous"]
    assert set("ACE") <= set(geocode.complex_lines(res["station"]["id"]))


def test_resolve_ambiguous_station():
    # Four unrelated 125 Sts (1 / 2,3 / 4,5,6 / A,B,C,D) are a real choice.
    res = geocode.resolve_station("125 St")
    assert res["ambiguous"]
    assert len(res["candidates"]) >= 3
    served = {tuple(geocode.STATION_BY_ID[c]["lines"]) for c in res["candidates"]}
    assert len(served) == len(res["candidates"])  # each candidate is a distinct station


def test_resolve_rejects_junk():
    with pytest.raises(geocode.GeocodeError):
        geocode.resolve_station("Atlantis")
    with pytest.raises(geocode.GeocodeError):
        geocode.resolve_station("Narnia")


def test_nearest_station_to_coordinate():
    near = geocode.nearest_station(40.7496, -73.9879)  # Herald Sq coords
    assert near["name"] == "34 St-Herald Sq"
    assert near["distance_km"] < 0.2


def test_geocode_landmark_to_nearest_station():
    g = geocode.geocode_place("Empire State Building")
    assert g["source"] == "landmark"
    assert g["nearest_station"]["name"] == "34 St-Herald Sq"


def test_geocode_fuzzy_landmark():
    g = geocode.geocode_place("empire state")
    assert g["matched"] == "empire state building"


def test_haversine_known_values():
    assert geocode.haversine_km(40.0, -73.0, 40.0, -73.0) == 0.0
    # ~1 degree of latitude ≈ 111 km
    assert 110 < geocode.haversine_km(40.0, -73.0, 41.0, -73.0) < 112
