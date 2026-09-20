"""
One-shot MinIO replay.

Reprocesses raw objects stored in MinIO and reinserts them into the DB.
Not part of the continuous scraper loop.
"""

import asyncio
import argparse

from logger import logger
from scraper import Scraper
from utils import load_configs, safe_emit
from monitoring.client import emit_event, emit_metric, emit_heartbeat
from raw_data.raw_storage import download_raw_data, list_raw_objects


class MinIOReplayScraper(Scraper):
    async def replay_from_minio(self, prefix: str = "", chunk_size: int = 500):
        """
        Reprocess all objects stored in MinIO and reinsert into DB.
        """
        logger.info(f"[{self.name}] Starting MinIO replay")
        safe_emit(
            emit_event,
            name="minio",
            instance_id="default",
            event_type="replay_started",
            severity="INFO",
            message=f"MinIO replay started for scraper {self.name}",
            metadata={"prefix": prefix},
        )

        object_names = list_raw_objects(prefix)

        logger.info(f"[{self.name}] Found {len(object_names)} raw objects")
        reprocessed = 0
        failed = 0

        for object_name in object_names:
            raw = download_raw_data(object_name)

            if not raw:
                logger.warning(f"Skipping unreadable object {object_name}")
                failed += 1
                continue
            try:
                extracted = self.extractor.extract(
                    raw,
                    self.scraper_config["root_tag"],
                )

                mapped = self.mapper.map_records(extracted)

                records = self.enricher.enrich_records(mapped)
                records = self.hash_records(records)

                unregistered = self.unregistered_records(records)
                self.register(unregistered)

                for i in range(0, len(records), chunk_size):
                    chunk = records[i : i + chunk_size]
                    self.send_measurements(chunk)

                logger.info(f"Reprocessed {object_name}")
                reprocessed += 1

            except Exception as e:
                logger.error(f"Replay failed for {object_name}: {e}")
                failed += 1
                safe_emit(
                    emit_event,
                    name="minio",
                    instance_id="default",
                    event_type="replay_object_failed",
                    severity="ERROR",
                    message=f"Replay failed for {object_name}: {e}",
                    metadata={"object_name": object_name},
                )

        safe_emit(
            emit_metric,
            name="minio",
            instance_id="default",
            metric_name="replay_objects_reprocessed",
            value=reprocessed,
            unit="count",
        )
        safe_emit(
            emit_metric,
            name="minio",
            instance_id="default",
            metric_name="replay_objects_failed",
            value=failed,
            unit="count",
        )
        safe_emit(
            emit_event,
            name="minio",
            instance_id="default",
            event_type="replay_completed",
            severity="INFO",
            message=f"MinIO replay completed for scraper {self.name}",
            metadata={"reprocessed": reprocessed, "failed": failed},
        )
        safe_emit(
            emit_heartbeat,
            name="minio",
            instance_id="default",
            status="OK" if failed == 0 else "FAIL",
        )


async def main():
    parser = argparse.ArgumentParser(description="Replay raw MinIO objects into the DB")
    parser.add_argument("--config", nargs="*", help="Specify which config(s) to use (none = all)")
    parser.add_argument("--prefix", default=None, help="MinIO object prefix filter (overrides config)")
    parser.add_argument("--chunk-size", type=int, default=500, help="Measurement batch size")
    args = parser.parse_args()

    # Replay uses continuous (non-CSV) configs by default.
    configs = load_configs(selected=args.config)

    logger.info("Starting MinIO replay")
    tasks = []
    for scraper_conf, mapping_conf in configs:
        scraper = MinIOReplayScraper(scraper_conf, mapping_conf)
        prefix = args.prefix if args.prefix is not None else scraper_conf.get("minio_prefix", "")
        tasks.append(scraper.replay_from_minio(prefix=prefix, chunk_size=args.chunk_size))

    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
