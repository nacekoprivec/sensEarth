"""
One-shot historic CSV import.

Reads a local CSV file, maps/enriches records, registers nodes/sensors,
and sends measurements in chunks. Not part of the continuous scraper loop.
"""

import asyncio
import argparse

from logger import logger
from scraper import Scraper
from utils import load_configs


class HistoricScraper(Scraper):
    async def run_historic(self, file_path: str = "ingest/data.csv"):
        """Processes a local file once and exits."""

        if self.format.lower() != "csv":
            return

        logger.info(f"Starting historic import for {file_path}")

        with open(file_path, "rb") as f:
            raw_data = f.read()

        # No fetcher here — file input only.
        _delimiter = self.scraper_config.get("root_tag", ";")
        extracted = self.extractor.extract(raw_data, _delimiter)

        try:
            self.mapper.validate_source_columns(
                records=extracted,
                headers=getattr(self.extractor, "fieldnames", None),
            )
        except ValueError as e:
            logger.error(f"[{self.name}] Aborting historic import: {e}")
            return

        mapped = self.mapper.map_records(extracted)

        records = self.enricher.enrich_records(mapped)
        records = self.hash_records(records)
        unregistered = self.unregistered_records(records)

        self.register(unregistered)

        inserted = []
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            inserted.append(self.send_measurements(chunk))
            logger.info(f"Progress: {i + len(chunk)}/{len(records)}")

        logger.info(f"Historic import completed. {inserted}")


async def main():
    parser = argparse.ArgumentParser(description="Historic CSV import")
    parser.add_argument("--config", nargs="*", help="Specify which config(s) to use (none = all)")
    parser.add_argument("--file", default="ingest/data.csv", help="Path to the CSV file to import")
    args = parser.parse_args()

    # Historic mode needs CSV configs (continuous scraper excludes them).
    configs = load_configs(selected=args.config, include_csv=True)

    tasks = []
    for scraper_conf, mapping_conf in configs:
        scraper = HistoricScraper(scraper_conf, mapping_conf)
        tasks.append(scraper.run_historic(file_path=args.file))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
