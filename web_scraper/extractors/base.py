
class Extractor:
    """Base class for all extractors."""
    def extract(self, data: bytes) -> list[dict]:
        raise NotImplementedError
