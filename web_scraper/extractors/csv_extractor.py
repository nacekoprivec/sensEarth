import csv
import io
from .base import Extractor


class CSVExtractor(Extractor):
    def __init__(self):
        self.fieldnames = None

    def extract(self, data: bytes, root_tag: str = ";") -> list[dict]:
        """
        data: Raw bytes from fetcher.
        root_tag: Used here as the delimiter (default is semicolon).
        """
        result = []
        self.fieldnames = None
        delimiter = root_tag or ";"

        try:
            content = self._decode(data)
            f = io.StringIO(content)

            reader = csv.DictReader(f, delimiter=delimiter)
            self.fieldnames = list(reader.fieldnames or [])

            for row in reader:
                clean_row = {
                    k.strip(): (v or "").strip()
                    for k, v in row.items()
                    if k is not None
                }
                result.append(clean_row)

        except Exception as e:
            print(f"[CSVExtractor] Error parsing CSV: {e}")

        return result
