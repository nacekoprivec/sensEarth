import os
import json
import asyncio

from fetcher import Fetcher
from mapper import Mapper
from extractors.xml_extractor import XMLExtractor
# from extractors.json_extractor import JSONExtractor
# from extractors.csv_extractor import CSVExtractor
# from extractors.html_extractor import HTMLExtractor

EXTRACTOR_MAP = {
    "xml": XMLExtractor,
    # "json": JSONExtractor,
    # "csv": CSVExtractor,
    # "html": HTMLExtractor
}

class Scraper:
    def __init__(self, scraper_config: dict, mapping_config: dict):
        self.scraper_config = scraper_config
        self.mapping_config = mapping_config

        self.fetcher = Fetcher()
        self.mapper = Mapper(mapping_config)

        format = scraper_config.get("format")

        if format not in EXTRACTOR_MAP:
            raise ValueError(f"Unsupported format: {format}")
        
        if format == "html":
            self.extractor = EXTRACTOR_MAP[format](selector=scraper_config.get("selector"))
        else:
            self.extractor = EXTRACTOR_MAP[format]()

        self.fetch_interval = scraper_config.get("fetch_interval", 0)
        self.name = scraper_config.get("name", "Unnamed Scraper")
        self.limit_records = scraper_config.get("limit_records", None)

    def run_once(self):
        raw = self.fetcher.fetch(self.scraper_config["target_url"])
        extracted = self.extractor.extract(raw)
        mapped = self.mapper.map_records(extracted)
 
        return mapped

    async def run(self):
        while True:
            try:
                records = self.run_once()
                records = records[: self.limit_records]
                print(f"\n[{self.name}] Mapped records:")
                for r in records:
                    print(r)
                print(f"[{self.name}] Total records fetched and mapped: {len(records)}\n")
            except Exception as e:
                print(f"[{self.name}] Error during scraping: {e}")

            if self.fetch_interval == 0:
                break
            await asyncio.sleep(self.fetch_interval)


def load_configs(folder="configs", selected=None):
    configs = []
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue
        name = os.path.splitext(file)[0]
        if selected and name not in selected:
            continue
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)
            configs.append((data["scraper_config"], data["mapping_config"]))
    return configs


async def main():
    configs = load_configs()
    scrapers = [Scraper(scraper_conf, mapping_conf) for scraper_conf, mapping_conf in configs]
    await asyncio.gather(*(s.run() for s in scrapers))


if __name__ == "__main__":
    asyncio.run(main())
