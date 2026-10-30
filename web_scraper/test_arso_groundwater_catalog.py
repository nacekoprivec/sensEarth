"""Unit tests for ARSO groundwater catalog."""

from unittest.mock import MagicMock, patch

from arso_groundwater.catalog import (
    display_name_from_label,
    fetch_layer_23_geojson,
    parse_station_feature,
    station_id_from_sifra,
    stations_from_feature_collection,
    validate_layer_23_geojson,
)


def _point_feature(
    sifra: str,
    lon: float = 14.5,
    lat: float = 46.0,
    ime: str = "Preserje (0430)",
    objectid: int = 1,
) -> dict:
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lon, lat]},
        "properties": {
            "OBJECTID": objectid,
            "SIFRA_MM": sifra,
            "IME_MM": ime,
            "LOKACIJA": "Dolina Kamniške Bistrice",
            "VTPODV_ID": "VTPodV_1001",
        },
    }


def _collection(*features: dict, exceeded: bool = False) -> dict:
    body: dict = {"type": "FeatureCollection", "features": list(features)}
    if exceeded:
        body["exceededTransferLimit"] = True
    return body


# --- station_id / name helpers ---


def test_station_id_keeps_leading_zeros():
    assert station_id_from_sifra("01094") == "01094"
    assert station_id_from_sifra("65020") == "65020"
    assert station_id_from_sifra("  65020  ") == "65020"
    assert station_id_from_sifra("") is None
    assert station_id_from_sifra(None) is None


def test_station_id_rejects_numeric_json_values():
    # Leading zeros cannot be recovered from int/float — must stay None.
    assert station_id_from_sifra(1094) is None
    assert station_id_from_sifra(1094.0) is None
    assert station_id_from_sifra(True) is None


def test_display_name_strips_only_trailing_paren():
    assert display_name_from_label("Preserje (0430)") == "Preserje"
    assert display_name_from_label("Foo (bar) (0430)") == "Foo (bar)"
    assert display_name_from_label("NoCode") == "NoCode"


# --- parse_station_feature ---


def test_parse_station_feature_happy_path():
    station, reason = parse_station_feature(
        _point_feature(
            "65020",
            lon=14.5943658314202,
            lat=46.1682065302387,
            ime="Preserje (0430)",
            objectid=103,
        )
    )
    assert reason is None
    assert station is not None
    assert station["station_id"] == "65020"
    assert station["name"] == "Preserje"
    assert station["archive_label"] == "Preserje (0430)"
    assert station["aquifer"] == "Dolina Kamniške Bistrice"
    assert station["longitude"] == 14.5943658314202
    assert station["latitude"] == 46.1682065302387
    assert station["water_body_id"] == "VTPodV_1001"
    assert station["gis_objectid"] == 103


def test_parse_station_feature_keeps_leading_zero_id():
    station, reason = parse_station_feature(
        _point_feature("01094", lon=16.1, lat=46.5, ime="Benica (Ben - 2/14)")
    )
    assert reason is None
    assert station is not None
    assert station["station_id"] == "01094"
    assert station["name"] == "Benica"


def test_parse_station_feature_rejects_missing_sifra():
    feature = _point_feature("65020")
    del feature["properties"]["SIFRA_MM"]
    station, reason = parse_station_feature(feature)
    assert station is None
    assert reason is not None
    assert "SIFRA_MM" in reason


def test_parse_station_feature_rejects_numeric_sifra():
    feature = _point_feature("65020")
    feature["properties"]["SIFRA_MM"] = 1094
    station, reason = parse_station_feature(feature)
    assert station is None
    assert reason is not None
    assert "numeric" in reason


def test_parse_station_feature_rejects_non_point():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": []},
        "properties": {"SIFRA_MM": "65020"},
    }
    station, reason = parse_station_feature(feature)
    assert station is None
    assert reason is not None
    assert "Point" in reason


def test_parse_station_feature_rejects_coords_outside_slovenia():
    # D96-metre-like values if outSR=4326 was ignored
    station, reason = parse_station_feature(
        _point_feature("65020", lon=468677.0, lat=114351.0)
    )
    assert station is None
    assert reason is not None
    assert "outside Slovenia" in reason


# --- validate_layer_23_geojson ---


def test_validate_rejects_html_maintenance_page():
    try:
        validate_layer_23_geojson(
            {},
            content_type="text/html; charset=UTF-8",
            raw_preview="\n\n<html><body>GIS portal</body></html>",
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "HTML" in str(exc)


def test_validate_rejects_exceeded_transfer_limit():
    try:
        validate_layer_23_geojson(_collection(_point_feature("65020"), exceeded=True))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "exceededTransferLimit" in str(exc)


def test_validate_rejects_wrong_type():
    try:
        validate_layer_23_geojson({"type": "Feature", "features": []})
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "FeatureCollection" in str(exc)


# --- stations_from_feature_collection ---


def test_stations_from_collection_happy_path():
    body = _collection(
        _point_feature("65020", objectid=1),
        _point_feature("01094", lon=16.1, lat=46.5, ime="Benica (Ben - 2/14)", objectid=2),
    )
    stations, report = stations_from_feature_collection(body, source="fixture")
    assert report["feature_count"] == 2
    assert report["station_count"] == 2
    assert report["skipped_count"] == 0
    assert [s["station_id"] for s in stations] == ["01094", "65020"]  # sorted


def test_stations_from_collection_hard_fails_on_duplicate_sifra():
    body = _collection(
        _point_feature("65020", objectid=1),
        _point_feature("65020", objectid=2, ime="Preserje duplicate"),
    )
    try:
        stations_from_feature_collection(body)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Duplicate SIFRA_MM" in str(exc)


def test_stations_from_collection_hard_fails_when_empty():
    bad = _point_feature("65020", lon=468677.0, lat=114351.0)  # out of bounds → skip
    body = _collection(bad)
    try:
        stations_from_feature_collection(body)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "zero usable stations" in str(exc)


def test_stations_from_collection_records_skip_reasons():
    body = _collection(
        _point_feature("65020"),
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [14.5, 46.0]},
            "properties": {"SIFRA_MM": 1094, "OBJECTID": 9},
        },
    )
    stations, report = stations_from_feature_collection(body)
    assert report["station_count"] == 1
    assert report["skipped_count"] == 1
    assert "numeric" in report["skipped"][0]["reason"]


# --- fetch_layer_23_geojson (mocked HTTP) ---


def test_fetch_raises_on_html_http_200():
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "text/html; charset=UTF-8"}
    mock_response.content = b"\n\n<html><h1>GIS portal ARSO</h1></html>\n"
    mock_response.json.side_effect = ValueError("No JSON")

    with patch(
        "arso_groundwater.catalog.request_with_retry",
        return_value=mock_response,
    ):
        try:
            fetch_layer_23_geojson()
            assert False, "expected RuntimeError"
        except RuntimeError as exc:
            assert "HTML" in str(exc)


def test_fetch_returns_validated_geojson():
    body = _collection(_point_feature("65020"))
    mock_response = MagicMock()
    mock_response.headers = {"content-type": "application/geo+json"}
    mock_response.content = b'{"type":"FeatureCollection","features":[]}'
    mock_response.json.return_value = body

    with patch(
        "arso_groundwater.catalog.request_with_retry",
        return_value=mock_response,
    ):
        result = fetch_layer_23_geojson()
    assert result["type"] == "FeatureCollection"
    assert len(result["features"]) == 1
    

if __name__ == "__main__":
    test_station_id_keeps_leading_zeros()
    test_station_id_rejects_numeric_json_values()
    test_display_name_strips_only_trailing_paren()
    test_parse_station_feature_happy_path()
    test_parse_station_feature_keeps_leading_zero_id()
    test_parse_station_feature_rejects_missing_sifra()
    test_parse_station_feature_rejects_numeric_sifra()
    test_parse_station_feature_rejects_non_point()
    test_parse_station_feature_rejects_coords_outside_slovenia()
    test_validate_rejects_html_maintenance_page()
    test_validate_rejects_exceeded_transfer_limit()
    test_validate_rejects_wrong_type()
    test_stations_from_collection_happy_path()
    test_stations_from_collection_hard_fails_on_duplicate_sifra()
    test_stations_from_collection_hard_fails_when_empty()
    test_stations_from_collection_records_skip_reasons()
    test_fetch_raises_on_html_http_200()
    test_fetch_returns_validated_geojson()
    print("ok")
