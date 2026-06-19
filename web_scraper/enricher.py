import math

from logger import logger

_NULL_STRINGS = frozenset({"null", "none", "n/a"})
_ALTITUDE_SENTINEL = "kota_0" # legacy sentinel for altitude


def clean_placeholder(value):
    """
    Normalize text fields: empty or sentinel strings become None.

    - None stays None
    - Empty or whitespace-only strings -> None
    - "null", "none", "n/a" (any case) -> None
    - Other strings -> stripped string
    - Non-string scalars pass through unchanged
    """
    if value is None:
        return None

    if not isinstance(value, str):
        return value

    stripped = value.strip()
    if stripped == "" or stripped.lower() in _NULL_STRINGS:
        return None

    return stripped


def coerce_numeric(value):
    """
    Convert a scalar measurement or coordinate value to float or None.

    - None stays None
    - bool -> None (avoid True/False silently becoming 1.0/0.0)
    - Empty or whitespace-only strings -> None
    - "null", "none", "n/a" (any case) -> None
    - "kota_0" (altitude sentinel) -> 0.0
    - Valid finite numeric strings and int/float -> float
    - nan / inf (would serialize to invalid JSON) -> None
    - Other types or unparseable strings -> None
    """
    if value is None:
        return None

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return _finite_or_none(float(value))

    if isinstance(value, str):
        cleaned = clean_placeholder(value)
        if cleaned is None:
            return None
        if cleaned.lower() == _ALTITUDE_SENTINEL:
            return 0.0
        try:
            parsed = float(cleaned)
        except ValueError:
            logger.warning(f"[Enricher] Could not parse numeric value: {value!r}")
            return None
        return _finite_or_none(parsed)

    return None


def _finite_or_none(number: float):
    """nan/inf serialize to invalid JSON (NaN/Infinity), so drop them to None."""
    return number if math.isfinite(number) else None
    

_COORD_FIELDS = ("longitude", "latitude", "altitude")
_NODE_TEXT_FIELDS = ("node_label", "node_serial")
_SENSOR_TEXT_FIELDS = ("sensor_label", "sensor_name", "sensor_description")


class Enricher:
    """
    Cleans and normalizes mapped records before hashing and ingest.

    Expects the Mapper output shape: {"node": {...}, "sensors": [...]}.
    """

    def clean_text_fields(self, entity: dict, fields: tuple[str, ...]) -> None:
        for field in fields:
            if field in entity:
                entity[field] = clean_placeholder(entity[field])

    def clean_coords(self, entity: dict) -> None:
        for field in _COORD_FIELDS:
            if field in entity:
                entity[field] = coerce_numeric(entity[field])

    def default_altitude(self, entity: dict) -> None:
        """API expects a numeric altitude; missing or cleaned-null becomes 0.0."""
        if entity.get("altitude") is None:
            entity["altitude"] = 0.0

    def clean_node(self, node: dict) -> None:
        self._clean_text_fields(node, _NODE_TEXT_FIELDS)
        self._clean_coords(node)
        self._default_altitude(node)

    def clean_measurements(self, sensor: dict) -> None:
        """
        Coerce measurement values to float or None.

        Modifies sensor["measurements"] in place. Timestamps are left
        unchanged; send_measurements handles those via normalize_timestamp.
        """
        measurements = sensor.get("measurements")
        if not isinstance(measurements, list):
            return

        for measurement in measurements:
            if not isinstance(measurement, dict):
                continue
            measurement["value"] = coerce_numeric(measurement.get("value"))

    def clean_sensor(self, sensor: dict) -> None:
        self._clean_text_fields(sensor, _SENSOR_TEXT_FIELDS)
        self._clean_coords(sensor)
        self._default_altitude(sensor)
        self._clean_measurements(sensor)

    def enrich_record(self, record: dict) -> dict:
        """
        Clean a single mapped record.

        Returns the same dict (in-place updates).
        """
        if not isinstance(record, dict):
            return record

        node = record.get("node")
        if isinstance(node, dict):
            self._clean_node(node)

        for sensor in record.get("sensors") or []:
            if isinstance(sensor, dict):
                self._clean_sensor(sensor)
        return record

    def enrich_records(self, records: list[dict]) -> list[dict]:
        return [self.enrich_record(r) for r in records]
