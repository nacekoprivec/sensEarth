"""
Parse ARSO pov_arhiv daily CSV and map columns to registered sensors.

Unit conversion: convert archive column unit → registered sensor.unit.
Timestamps are date-only and stored as calendar midnight UTC (no local tz shift).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from extractors.csv_extractor import CSVExtractor

# Archive column name (exact header) → SensEarth sensor_type name
COLUMN_TO_SENSOR_TYPE = {
    "vodostaj (cm)": "water_level",
    "pretok (m3/s)": "water_flow",
    "temp. vode (°C)": "water_temperature",
    "temp. vode (°c)": "water_temperature",
}

# Factors: multiply archive value by factor to get target unit
# key = (from_unit, to_unit)
UNIT_FACTORS = {
    ("cm", "m"): 0.01,
    ("m", "cm"): 100.0,
    ("mm", "m"): 0.001,
    ("m", "mm"): 1000.0,
    ("mm", "cm"): 0.1,
    ("cm", "mm"): 10.0,
}


def normalize_unit(unit: Optional[str]) -> Optional[str]:
    if unit is None:
        return None
    text = str(unit).strip().lower()
    text = text.replace("°", "")
    text = text.replace(" ", "")
    aliases = {
        "c": "c",
        "°c": "c",
        "m3/s": "m3/s",
        "m^3/s": "m3/s",
        "m³/s": "m3/s",
    }
    return aliases.get(text, text)


def unit_from_header(column: str) -> Optional[str]:
    """Extract unit from ARSO header like 'vodostaj (cm)'."""
    match = re.search(r"\(([^)]+)\)\s*$", column.strip())
    if not match:
        return None
    return normalize_unit(match.group(1))


def conversion_factor(from_unit: Optional[str], to_unit: Optional[str]) -> Tuple[float, str]:
    """
    Return (factor, note). factor=1.0 means no conversion.

    Both archive and registered units are required. Missing either raises
    ValueError so historic import never silently assumes a scale.
    """
    src = normalize_unit(from_unit)
    dst = normalize_unit(to_unit)

    if not src:
        raise ValueError(f"Archive column unit is missing (from_unit={from_unit!r})")
    if not dst:
        raise ValueError(
            f"Registered sensor.unit is missing (to_unit={to_unit!r}). "
            "GET /sensors must return unit from sensor_type."
        )

    if src == dst:
        return 1.0, f"same unit ({src})"

    factor = UNIT_FACTORS.get((src, dst))
    if factor is None:
        raise ValueError(f"No conversion from '{from_unit}' to '{to_unit}'")

    return factor, f"convert {src} -> {dst} (x{factor})"


def parse_date_to_utc_midnight(date_text: str) -> str:
    """
    Parse DD.MM.YYYY (or with time) to 'YYYY-MM-DD 00:00:00'.

    Date-only archive values keep the calendar day; no local→UTC shift.
    """
    raw = (date_text or "").strip()
    if not raw:
        raise ValueError("empty date")

    for fmt in ("%d.%m.%Y", "%d.%m.%Y %H:%M", "%d.%m.%Y %H:%M:%S"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.strftime("%Y-%m-%d 00:00:00")
        except ValueError:
            continue
    raise ValueError(f"unrecognized date: {date_text!r}")


def map_headers_to_types(headers: List[str]) -> Dict[str, str]:
    """
    Map present CSV headers to sensor_type names.
    Unknown columns are ignored.
    """
    mapping: Dict[str, str] = {}
    for header in headers:
        key = header.strip()
        lowered = {k.lower(): v for k, v in COLUMN_TO_SENSOR_TYPE.items()}
        sensor_type = COLUMN_TO_SENSOR_TYPE.get(key) or lowered.get(key.lower())
        if sensor_type:
            mapping[key] = sensor_type
    return mapping


def parse_daily_csv(raw: bytes, sensors_by_type: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:  
    """
    Parse ARSO daily CSV bytes into measurement candidates.

    sensors_by_type: { sensor_type: sensor_dict from GET /sensors }

    Returns a report dict:
      {
        "headers": [...],
        "column_map": {column: sensor_type},
        "conversions": {sensor_type: note},
        "warnings": [...],
        "measurements": [{sensor_hash, timestamp_utc, value, source_column, sensor_type}],
        "stats": {seen, kept, skipped_empty, skipped_bad_ts, skipped_bad_value, skipped_no_sensor}
      }
    """
    extractor = CSVExtractor()
    rows = extractor.extract(raw, ";")
    headers = [h.strip() for h in (extractor.fieldnames or []) if h]

    column_map = map_headers_to_types(headers)
    warnings: List[str] = []
    conversions: Dict[str, str] = {}

    if not column_map:
        raise ValueError(
            f"No recognized ARSO daily columns in headers={headers}. "
            f"Expected one of {sorted(COLUMN_TO_SENSOR_TYPE)}"
        )

    # Build per-column conversion using registered sensor.unit from GET /sensors.
    # Skip the column (with warning) when unit is missing or conversion unknown.
    factors: Dict[str, float] = {}
    blocked_columns: set[str] = set()
    for column, sensor_type in column_map.items():
        sensor = sensors_by_type.get(sensor_type)
        if not sensor:
            warnings.append(f"CSV column {column!r} → {sensor_type}: no registered sensor; skip")
            continue

        if not sensor.get("sensor_hash"):
            warnings.append(f"{sensor_type}: sensor_hash missing from API; column skipped")
            continue

        archive_unit = unit_from_header(column)
        registered_unit = sensor.get("unit")
        try:
            factor, note = conversion_factor(archive_unit, registered_unit)
        except ValueError as exc:
            warnings.append(f"{sensor_type}: {exc}; column skipped")
            continue

        # Block water_level when archive is cm but registered unit is m.
        # Live ARSO XML vodostaj is typically cm-scale while sensor_type.unit
        # is often "m"; converting historic cm->m would disagree with live data.
        if (
            sensor_type == "water_level"
            and normalize_unit(registered_unit) == "m"
            and normalize_unit(archive_unit) == "cm"
            and factor == 0.01
        ):
            warnings.append(
                "BLOCKED water_level: registered unit is 'm' but archive column "
                "is cm (live XML vodostaj is typically cm-scale). Fix "
                "sensor_type.unit to match stored live values before ingesting "
                "water_level. Flow/temp still allowed."
            )
            blocked_columns.add(column)
            continue

        factors[column] = factor
        conversions[sensor_type] = (
            f"archive={archive_unit} registered={registered_unit} ({note})"
        )

    measurements: List[Dict[str, Any]] = []
    stats = {
        "seen": 0,
        "kept": 0,
        "skipped_empty": 0,
        "skipped_bad_ts": 0,
        "skipped_bad_value": 0,
        "skipped_no_sensor": 0,
        "skipped_out_of_range": 0,
        "skipped_blocked_unit": 0,
    }

    for row in rows:
        date_raw = (row.get("Datum") or "").strip()
        if not date_raw:
            continue

        for column, sensor_type in column_map.items():
            stats["seen"] += 1
            sensor = sensors_by_type.get(sensor_type)
            if not sensor:
                stats["skipped_no_sensor"] += 1
                continue
            if column in blocked_columns or column not in factors:
                if column in blocked_columns:
                    stats["skipped_blocked_unit"] += 1
                else:
                    stats["skipped_no_sensor"] += 1
                continue

            raw_value = (row.get(column) or "").strip()
            if raw_value == "":
                stats["skipped_empty"] += 1
                continue

            try:
                value = float(raw_value.replace(",", ".")) * factors[column]
            except ValueError:
                stats["skipped_bad_value"] += 1
                continue

            try:
                ts = parse_date_to_utc_midnight(date_raw)
            except ValueError:
                stats["skipped_bad_ts"] += 1
                continue

            vmin = sensor.get("value_min")
            vmax = sensor.get("value_max")
            if vmin is not None and value < float(vmin):
                stats["skipped_out_of_range"] += 1
                continue
            if vmax is not None and value > float(vmax):
                stats["skipped_out_of_range"] += 1
                continue

            measurements.append(
                {
                    "sensor_hash": sensor["sensor_hash"],
                    "timestamp_utc": ts,
                    "value": value,
                    "source_column": column,
                    "sensor_type": sensor_type,
                    "sensor_id": sensor.get("sensor_id"),
                }
            )
            stats["kept"] += 1

    return {
        "headers": headers,
        "column_map": column_map,
        "conversions": conversions,
        "warnings": warnings,
        "measurements": measurements,
        "stats": stats,
    }
