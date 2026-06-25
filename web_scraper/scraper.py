from typing import Dict, Any, List
import hashlib
import os
import json
import asyncio
import argparse
import requests
import time

import traceback 
from logger import logger
from fetcher import Fetcher
from mapper import Mapper
from enricher import Enricher
from utils import *
from extractors.xml_extractor import XMLExtractor
from extractors.json_extractor import JSONExtractor
from extractors.csv_extractor import CSVExtractor
from extractors.html_extractor import HTMLExtractor

from monitoring.client import emit_component_registration, emit_event, emit_metric, emit_heartbeat

from raw_data.raw_storage import download_raw_data, list_raw_objects, MINIO_INSTANCE_ID

EXTRACTOR_MAP = {
    "xml": XMLExtractor,
    "json": JSONExtractor,
    "csv": CSVExtractor,
    "html": HTMLExtractor
}

API_URL = os.getenv("MIDDLEWARE_API")
STATE_DIR = "state"

os.makedirs(STATE_DIR, exist_ok=True)

class Scraper:
    def __init__(self, scraper_config: dict, mapping_config: dict):
        """
        Fetcher is responsible for fetching raw data from target URL.
        Extractor is responsible for extracting records from raw data based on format.
        Mapper is responsible for mapping extracted data to the required format.
        Enricher is responsible for cleaning and normalizing mapped records.
        State is used to track registered nodes/sensors in JSON file as hash->id mapping.
        """
        self.scraper_config = scraper_config
        self.mapping_config = mapping_config

        self.fetcher = Fetcher()
        self.mapper = Mapper(mapping_config)
        self.enricher = Enricher()

        self.format = scraper_config.get("format")
        if self.format not in EXTRACTOR_MAP:
            raise ValueError(f"Unsupported format: {self.format}")
        else:
            self.extractor = EXTRACTOR_MAP[self.format]()

        self.fetch_interval = scraper_config.get("fetch_interval", 0)
        self.name = scraper_config.get("name", "Unnamed Scraper")
        self.limit_results = scraper_config.get("limit_results", None)

        # Load or initialize state
        self.state_file = os.path.join(STATE_DIR, f"{self.name}_state.json")
        self.state = self.load_state()

        safe_emit(emit_component_registration, name="scraper", instance_id=self.name, component_type="scraper")
        safe_emit(emit_component_registration, name="minio", instance_id=MINIO_INSTANCE_ID, component_type="minio")
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="OK")
        safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="bucket_ready", severity="INFO", message=f"MinIO ready for scraper {self.name}")

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(self.state, f, indent=2)

    def load_state(self):
        """
        Loads file state. It is located in docker container.
        Contains pairs of "nodes": { node_hash : node_id}, "sensors": { sensor_hash : sensor_id}}
        """
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
            else:
                state = {}

            # Backward-compatible defaults
            if not isinstance(state, dict):
                state = {}
            state.setdefault("nodes", {})
            state.setdefault("sensors", {})
            return state
        except json.JSONDecodeError as e:
            logger.error(f"Error loading state for {self.name}: {e}")
            return {"nodes": {}, "sensors": {}}

    def register(self, payload: Dict) -> Dict:
        """
        Registers nodes and sensors from the payload using the /register endpoint.
        Returns pairs of "nodes": { node_hash : node_id}, "sensors": { sensor_hash : sensor_id}}
        """
        normalize(payload)

        if not payload.get("nodes") and not payload.get("sensors"):
            logger.info(f"Nothing to register")
            return {} 
        try:
            response = retry_request(
                requests.post,
                retries=5,
                delay=5,
                backoff=2,
                url=f"{API_URL}/register",
                json=payload
            )
            data = response.json()
            self.state["nodes"].update(data.get("nodes", {}))
            self.state["sensors"].update(data.get("sensors", {}))
            self.save_state()

            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="registration_success",severity="INFO",message=f"Registered {len(data.get('nodes', {}))} nodes and {len(data.get('sensors', {}))} sensors")
            return data
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="registration_failure",severity="ERROR",message=f"Registration failed | nodes={len(payload.get('nodes', []))} sensors={len(payload.get('sensors', []))} | error={e}")
            return {}

    def send_measurements(self, payload: List[Dict]):
        """
        Sends measurements to the API.
        Expects payload with sensors and their measurements.
        Skips unknown sensors and invalid timestamps, normalizes timestamps,
        and sends valid data to the `/dataIngest` endpoint.
        Emits metrics for sent and skipped measurements and logs success/failure.
        """
        measurements = []
        skipped = 0 # track skipped sensors
        for entry in payload:
            for sensor in entry.get("sensors", []):
                sensor_hash = sensor["sensor_hash"]
                sensor_id = self.state["sensors"].get(sensor_hash)
                if not sensor_id:
                    skipped += 1
                    logger.warning(f"Skipping unknown sensor {sensor_hash}")
                    continue
                for m in sensor.get("measurements", []):
                    try:
                        ts = m["timestamp_utc"]
                        normalized_ts = normalize_timestamp(ts)
                    except ValueError:
                        logger.warning(f"Skipping invalid timestamp: {ts}")
                        continue
                    measurements.append({
                        "sensor_hash": sensor_hash,
                        "timestamp_utc": normalized_ts,
                        "value": m["value"]
                    })

        if measurements:
            try:
                response = retry_request(
                    requests.post,
                    retries=5,
                    delay=5,
                    backoff=2,
                    url=f"{API_URL}/dataIngest",
                    json=measurements
                )

                safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="data_ingest_success",severity="INFO",message=f"Sent measurements successfully")
                if len(measurements) > 0:
                   skipped_rate = (skipped / len(measurements)) * 100
                else:
                   skipped_rate = 0
                safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="measurements_skipped_rate", value=skipped_rate)

                return response.json()
            except Exception as e:  
                logger.error(f"Error sending measurements: {traceback.format_exc()}")
                safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="data_ingest_failure",severity="ERROR",message=f"Failed to send measurements: {e}")
        return {}
    
    def stable_hash(self, obj) -> str:
        """
        Create a stable hash based on sorted JSON representation.
        Ensures same structure gives same hash every run.
        """
        dumped = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(dumped.encode()).hexdigest()

    def hash_records(self, records: List[Dict]) -> List[Dict]:
        """
        Hash nodes and sensors in records if not already present.
        """
        for record in records:
            node = record.get("node", {})
            if node.get("node_hash") is None:
                hash_fields = self.scraper_config.get("node_hash_fields", [])
                node_hash_input = {field: node.get(field) for field in hash_fields}
                node["node_hash"] = self.stable_hash(node_hash_input)

            for sensor in record.get("sensors", []):
                if "sensor_hash" not in sensor:
                    st_name = sensor.get("sensor_type", {}).get("name")
                    hash_fields = self.scraper_config.get("sensor_hash_fields", [])
                    sensor_hash_input = {field: sensor.get(field) for field in hash_fields}
                    sensor["sensor_hash"] = self.stable_hash({
                        "node_hash": node["node_hash"],
                        "sensor_type": st_name,
                        "longitude": sensor.get("longitude"),
                        "latitude": sensor.get("latitude"),
                        "altitude": sensor.get("altitude")
                    })
        return records

    def unregistered_records(self, records: List[Dict]) -> List[Dict]:
        """
        Identifies records with unregistered nodes/sensors.
        Returns only dictionaries for nodes/sensors that are not yet registered according to the state.
        """
        to_register = {"nodes": [], "sensors": []}
        for record in records:
            node = record.get("node")
            if node:
                node_hash = node["node_hash"]
                if node_hash not in self.state["nodes"]:
                    to_register["nodes"].append(node)

            for sensor in record.get("sensors", []):
                sensor_hash = sensor["sensor_hash"]
                if sensor_hash not in self.state["sensors"]:
                    sensor_entry = sensor.copy()
                    if node:
                        sensor_entry["node_hash"] = node["node_hash"]
                    to_register["sensors"].append(sensor_entry)

        return to_register

    def run_once(self):
        loop_start = time.time()
        try: 
            safe_emit(emit_heartbeat, name="scraper", instance_id=self.name, status="OK")
            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="scrape_started",severity="INFO",message="Scraping cycle started")
            
            fetch_result = self.fetcher.fetch(self.scraper_config["target_url"])

            raw = fetch_result["content"]
            is_new = fetch_result["is_new"]
            object_name = fetch_result["object_name"]
            safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="fetch_raw_duration_seconds", value=time.time() - loop_start)

            if not is_new: # If minio content is duplicated, skip processing 
                logger.info(f"[{self.name}] Duplicate raw skipped: {object_name}")
                safe_emit(emit_event, name="scraper", instance_id=self.name, event_type="duplicate_raw_skipped",severity="INFO", message=f"Skipped duplicate raw object {object_name}")
                safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="duplicate_raw_count", value=1)
                return []

            extracted = self.extractor.extract(raw, self.scraper_config["root_tag"])
            mapped = self.mapper.map_records(extracted)

            safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="scrape_duration_seconds", value=time.time() - loop_start)
            return mapped
        except Exception:
            tb = traceback.format_exc()
            logger.error(f"[{self.name}] Error during scraping run_once", exc_info=True, extra={"traceback": tb})
            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="scrape_failed",severity="ERROR",message=f"Scraping failed")
            safe_emit(emit_heartbeat, name="scraper", instance_id=self.name, status="FAIL")
            return []

    async def run(self):
        while True:
            try:
                #Scrape and map data
                records = self.run_once()
                records = records[: self.limit_results] if self.limit_results else records

                # Hash every record, register unregistered nodes/sensors and send all measurements.
                records = self.hash_records(records)
                records = self.enricher.enrich_records(records)
                unregistered = self.unregistered_records(records)
                self.register(unregistered)
                self.send_measurements(records)

                logger.info(f"[{self.name}] Total records processed: {len(records)}")

            except Exception as e:
                logger.error(f"[{self.name}] Error during scraping: {e}")
                safe_emit(emit_heartbeat, name="scraper", instance_id=self.name, status="FAIL")


            if self.fetch_interval <= 0:
                break
            await asyncio.sleep(self.fetch_interval)

class HistoricScraper(Scraper):
    async def run_historic(self, file_path: str = "ingest/data.csv"):
        """Processes a local file once and exits."""

        if self.format.lower() != 'csv': 
            return

        logger.info(f"Starting historic import for {file_path}")

        with open(file_path, "rb") as f:
            raw_data = f.read()

        # No fetcher, here.File input only.
        _delimiter = self.scraper_config.get("root_tag", ";") 
        extracted = self.extractor.extract(raw_data, _delimiter)
        mapped = self.mapper.map_records(extracted)

        records = self.hash_records(mapped)
        records = self.enricher.enrich_records(records)
        unregistered = self.unregistered_records(records)
        
        self.register(unregistered)

        inserted = []
        chunk_size = 500
        for i in range(0, len(records), chunk_size):
            chunk = records[i : i + chunk_size]
            inserted.append(self.send_measurements(chunk))
            logger.info(f"Progress: {i + len(chunk)}/{len(records)}")
        
        logger.info(f"Historic import completed. {inserted}")

class MinIOReplayScraper(Scraper):
    async def replay_from_minio(self, prefix: str = "", chunk_size: int = 500):
        """
        Reprocess all objects stored in MinIO and reinsert into DB.
        """

        logger.info(f"[{self.name}] Starting MinIO replay")
        safe_emit(emit_event, name="minio", instance_id="default", event_type="replay_started", severity="INFO", message=f"MinIO replay started for scraper {self.name}", metadata={"prefix": prefix})

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
                    self.scraper_config["root_tag"]
                )

                mapped = self.mapper.map_records(extracted)

                records = self.hash_records(mapped)
                records = self.enricher.enrich_records(records)

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
                safe_emit(emit_event, name="minio", instance_id="default", event_type="replay_object_failed", severity="ERROR", message=f"Replay failed for {object_name}: {e}", metadata={"object_name": object_name})

        safe_emit(emit_metric, name="minio", instance_id="default", metric_name="replay_objects_reprocessed", value=reprocessed, unit="count")
        safe_emit(emit_metric, name="minio", instance_id="default", metric_name="replay_objects_failed", value=failed, unit="count")
        safe_emit(emit_event, name="minio", instance_id="default", event_type="replay_completed", severity="INFO", message=f"MinIO replay completed for scraper {self.name}", metadata={"reprocessed": reprocessed, "failed": failed})
        safe_emit(emit_heartbeat, name="minio", instance_id="default", status="OK" if failed == 0 else "FAIL")


async def main():
    parser = argparse.ArgumentParser(description="Anomaly Detector CLI")

    parser.add_argument("--config", nargs="*", help="Specify which config(s) to use (none = all)")
    parser.add_argument("--historic", action="store_true", help="Run historic import")
    parser.add_argument("--minio_reinsert", action="store_true", help="Replay raw files stored in MinIO")

    args = parser.parse_args()
    configs = load_configs(selected=args.config)

    if args.historic:
        tasks = []
        for scraper_conf, mapping_conf in configs:
            scraper = HistoricScraper(scraper_conf, mapping_conf)
            tasks.append(scraper.run_historic())

        await asyncio.gather(*tasks)

    elif args.minio_reinsert:
        logger.info("Starting MinIO replay for all scrapers")
        tasks = []

        for scraper_conf, mapping_conf in configs:
            scraper = MinIOReplayScraper(scraper_conf, mapping_conf)
            tasks.append(scraper.replay_from_minio(prefix=scraper_conf.get("minio_prefix", "")))

        await asyncio.gather(*tasks)
        
    else:
        scrapers = [Scraper(scraper_conf, mapping_conf) for scraper_conf, mapping_conf in configs]
        await asyncio.gather(*(s.run() for s in scrapers))

if __name__ == "__main__":
    asyncio.run(main())
