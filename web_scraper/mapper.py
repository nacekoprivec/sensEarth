from typing import Optional


class Mapper:
    """
    Maps extracted records to DB schema based on mapping_config (supports nested dicts).
    Returns configuration value, if no record key matches. 
    For example, mapping_config = {"db_field": "record_key"} will map record["record_key"] to db_field.
    """

    def __init__(self, mapping_config: dict):
        self.mapping_config = mapping_config

    def _get_by_path(self, record, path: str):
        """
        Returns None if any segment is missing or traversal is impossible.
        """
        if not isinstance(record, dict):
            return None
        cur = record
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur.get(part)
        return cur

    def _map_value(self, config_val, record):
        # If nested mapping (dict), recurse
        if isinstance(config_val, dict):
            return {k: self._map_value(v, record) for k, v in config_val.items()}

        # If list template
        if isinstance(config_val, list):
            return [self._map_value(v, record) for v in config_val]

        # If config_val refers to a record key
        if isinstance(config_val, str):
            if config_val in record:
                return record.get(config_val)
            if "." in config_val:
                nested = self._get_by_path(record, config_val)
                if nested is not None:
                    return nested

        # Constant / fallback value
        return config_val

    def map_record(self, record: dict) -> dict:
        """
        Maps database fields to record values based on mapping_config.
        """
        return {
            db_key: self._map_value(config_value, record)
            for db_key, config_value in self.mapping_config.items()
        }

    def map_records(self, records: list[dict]) -> list[dict]:
        return [self.map_record(r) for r in records]

    def required_source_columns(self) -> set[str]:
        """
        Mapping strings that must exist as source columns/keys.
        Skips hardcoded node labels and other constants.
        """
        required: set[str] = set()

        node = self.mapping_config.get("node", {})
        node_serial = node.get("node_serial")
        if isinstance(node_serial, str):
            required.add(node_serial)

        for sensor in self.mapping_config.get("sensors", []):
            for measurement in sensor.get("measurements", []):
                for field in ("value", "timestamp_utc"):
                    column = measurement.get(field)
                    if isinstance(column, str):
                        required.add(column)

            metadata = sensor.get("metadata", {})
            if isinstance(metadata, dict):
                for column in metadata.values():
                    if isinstance(column, str):
                        required.add(column)

            for field in ("sensor_label", "longitude", "latitude", "altitude"):
                column = sensor.get(field)
                if isinstance(column, str):
                    required.add(column)

        return required

    def validate_source_columns(
        self,
        records: Optional[list[dict]] = None,
        headers: Optional[list[str]] = None,
    ) -> None:
        """
        Fail fast when mapped source columns are missing from CSV headers/rows.
        """
        required = self.required_source_columns()
        if not required:
            return

        available: set[str] = set()
        if headers:
            available.update(h.strip() for h in headers if h)
        if records:
            for row in records:
                if isinstance(row, dict):
                    available.update(row.keys())

        missing = sorted(required - available)
        if missing:
            raise ValueError(
                f"Missing required source columns: {missing}. "
                f"Available columns: {sorted(available)}"
            )

