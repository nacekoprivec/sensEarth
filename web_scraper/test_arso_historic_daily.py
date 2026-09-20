"""Unit tests for ARSO historic daily CSV parsing (no network)."""

from arso_historic.daily_csv import (
    conversion_factor,
    parse_daily_csv,
    parse_date_to_utc_midnight,
    unit_from_header,
)


SAMPLE_CSV = (
    "Datum;vodostaj (cm);pretok (m3/s)\r\n"
    "01.01.1994;76;0.309\r\n"
    "02.01.1994;;\r\n"
    "03.01.1994;74;0.256\r\n"
).encode("utf-8")


def test_unit_from_header():
    assert unit_from_header("vodostaj (cm)") == "cm"
    assert unit_from_header("pretok (m3/s)") == "m3/s"


def test_conversion_cm_to_m():
    factor, note = conversion_factor("cm", "m")
    assert factor == 0.01
    assert "cm" in note


def test_conversion_same_unit():
    factor, note = conversion_factor("m3/s", "m^3/s")
    assert factor == 1.0


def test_conversion_requires_registered_unit():
    try:
        conversion_factor("cm", None)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Registered sensor.unit is missing" in str(exc)


def test_conversion_requires_archive_unit():
    try:
        conversion_factor(None, "m")
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Archive column unit is missing" in str(exc)


def test_parse_date_keeps_calendar_day():
    assert parse_date_to_utc_midnight("01.01.1994") == "1994-01-01 00:00:00"


def test_parse_daily_csv_with_unit_conversion():
    sensors = {
        "water_level": {
            "sensor_hash": "hash_level",
            "sensor_id": 1,
            "unit": "m",
            "value_min": 0,
            "value_max": 1000,
        },
        "water_flow": {
            "sensor_hash": "hash_flow",
            "sensor_id": 2,
            "unit": "m^3/s",
            "value_min": 0,
            "value_max": 100,
        },
    }
    result = parse_daily_csv(SAMPLE_CSV, sensors)
    assert result["stats"]["skipped_empty"] >= 1
    assert result["stats"]["kept"] == 4  # 2 days × 2 sensors with values

    level_vals = [
        m["value"] for m in result["measurements"] if m["sensor_type"] == "water_level"
    ]
    assert abs(level_vals[0] - 0.76) < 1e-9  # 76 cm → 0.76 m
    assert any("cm → m" in w or "cm" in w for w in result["warnings"]) or result[
        "conversions"
    ].get("water_level")


def test_parse_daily_csv_no_convert_when_unit_cm():
    sensors = {
        "water_level": {
            "sensor_hash": "hash_level",
            "sensor_id": 1,
            "unit": "cm",
            "value_min": 0,
            "value_max": 1000,
        },
    }
    result = parse_daily_csv(SAMPLE_CSV, sensors)
    level_vals = [
        m["value"] for m in result["measurements"] if m["sensor_type"] == "water_level"
    ]
    assert level_vals[0] == 76.0


if __name__ == "__main__":
    test_unit_from_header()
    test_conversion_cm_to_m()
    test_conversion_same_unit()
    test_conversion_requires_registered_unit()
    test_conversion_requires_archive_unit()
    test_parse_date_keeps_calendar_day()
    test_parse_daily_csv_with_unit_conversion()
    test_parse_daily_csv_no_convert_when_unit_cm()
    print("ok")
