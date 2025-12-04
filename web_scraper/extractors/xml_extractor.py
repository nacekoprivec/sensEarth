from lxml import etree
from .base import Extractor

class XMLExtractor(Extractor):
    def extract(self, data: bytes) -> list[dict]:
        result = []
        try:
            root = etree.fromstring(data)
            stations = root.findall(".//postaja")
            for st in stations:
                rec = dict(st.attrib)
                for child in st:
                    rec[child.tag] = child.text
                result.append(rec)
        except Exception as e:
            print(f"[XMLExtractor] Error parsing XML: {e}")
        return result
