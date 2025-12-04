class Mapper:
    """Maps extracted records to DB schema based on mapping_config."""
    def __init__(self, mapping_config: dict):
        self.mapping_config = mapping_config

    def map_record(self, record: dict) -> dict:
        mapped = {}
        for db_key, config_value in self.mapping_config.items():
            if config_value is None:
                mapped[db_key] = None
            elif config_value in record:
                mapped[db_key] = record.get(config_value)
            else:
                mapped[db_key] = config_value
        return mapped

    def map_records(self, records: list[dict]) -> list[dict]:
        return [self.map_record(r) for r in records]
