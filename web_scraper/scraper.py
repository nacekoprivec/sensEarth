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

from raw_data.raw_storage import MINIO_INSTANCE_ID

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
        Contains:
          - nodes: { node_hash: node_id }
          - sensors: { sensor_hash: sensor_id }
          - sensor_metadata: { sensor_hash: { ... } }  # last metadata successfully registered
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
            state.setdefault("sensor_metadata", {})
            return state
        except json.JSONDecodeError as e:
            logger.error(f"Error loading state for {self.name}: {e}")
            return {"nodes": {}, "sensors": {}, "sensor_metadata": {}}

    def register(self, payload: Dict) -> Dict:
        """
        Registers nodes and sensors from the payload using the /register endpoint.
        Returns pairs of "nodes": { node_hash : node_id}, "sensors": { sensor_hash : sensor_id}}
        On success, also records sensor metadata in state so we do not re-upsert every cycle.
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

            # Remember metadata that was accepted so later cycles can skip no-op upserts
            registered_hashes = set(data.get("sensors", {}).keys())
            for sensor in payload.get("sensors", []):
                sensor_hash = sensor.get("sensor_hash")
                metadata = sensor.get("metadata")
                if sensor_hash in registered_hashes and metadata:
                    self.state.setdefault("sensor_metadata", {})[sensor_hash] = metadata

            self.save_state()

            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="registration_success",severity="INFO",message=f"Registered {len(data.get('nodes', {}))} nodes and {len(data.get('sensors', {}))} sensors")
            return data
        except Exception as e:
            logger.error(f"Error during registration: {e}")
            safe_emit(emit_event, name="scraper",instance_id=self.name,event_type="registration_failure",severity="ERROR",message=f"Registration failed | nodes={len(payload.get('nodes', []))} sensors={len(payload.get('sensors', []))} | error={e}")
            return {}

    def send_measurements(self, payload: List[Dict]):
        """
        Sends measurements to the API by sensor_hash.
        Middleware resolves hash → sensor_id; skips invalid timestamps only.
        """
        measurements = []
        skipped_invalid_ts = 0
        for entry in payload:
            for sensor in entry.get("sensors", []):
                sensor_hash = sensor["sensor_hash"]
                for m in sensor.get("measurements", []):
                    try:
                        ts = m["timestamp_utc"]
                        normalized_ts = normalize_timestamp(ts)
                    except ValueError:
                        skipped_invalid_ts += 1
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
                total = len(measurements) + skipped_invalid_ts
                skipped_rate = (skipped_invalid_ts / total) * 100 if total else 0
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
        Also re-sends sensors whose metadata differs from what was last
        successfully registered (e.g. missing metadata.sifra backfill).
        """
        to_register = {"nodes": [], "sensors": []}
        known_metadata = self.state.get("sensor_metadata", {})

        for record in records:
            node = record.get("node")
            if node:
                node_hash = node["node_hash"]
                if node_hash not in self.state["nodes"]:
                    to_register["nodes"].append(node)

            for sensor in record.get("sensors", []):
                sensor_hash = sensor["sensor_hash"]
                desired_metadata = sensor.get("metadata") or {}
                known = known_metadata.get(sensor_hash)

                needs_register = sensor_hash not in self.state["sensors"]
                needs_metadata_upsert = bool(desired_metadata) and desired_metadata != known

                if needs_register or needs_metadata_upsert:
                    sensor_entry = {
                        k: v for k, v in sensor.items() if k != "measurements"
                    }
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

            # Default: skip unchanged MinIO objects. Opt in via process_duplicate_raw
            # (temporary for ARSO hydro so metadata.sifra can backfill on existing sensors).
            process_duplicate_raw = bool(self.scraper_config.get("process_duplicate_raw", False))
            if not is_new and not process_duplicate_raw:
                logger.info(f"[{self.name}] Duplicate raw skipped: {object_name}")
                safe_emit(emit_event, name="scraper", instance_id=self.name, event_type="duplicate_raw_skipped",severity="INFO", message=f"Skipped duplicate raw object {object_name}")
                safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="duplicate_raw_count", value=1)
                return []
            if not is_new and process_duplicate_raw:
                logger.info(f"[{self.name}] Duplicate raw reprocessed (process_duplicate_raw=true): {object_name}")
                safe_emit(emit_event, name="scraper", instance_id=self.name, event_type="duplicate_raw_reprocessed", severity="INFO", message=f"Reprocessed duplicate raw object {object_name}")
                safe_emit(emit_metric, name="scraper", instance_id=self.name, metric_name="duplicate_raw_reprocessed_count", value=1)

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

    def _run_cycle(self):
        records = self.run_once()
        records = records[: self.limit_results] if self.limit_results else records

        records = self.enricher.enrich_records(records)
        records = self.hash_records(records)
        unregistered = self.unregistered_records(records)
        self.register(unregistered)
        self.send_measurements(records)

        logger.info(f"[{self.name}] Total records processed: {len(records)}")

    async def run(self):
        while True:
            try:
                # Blocking HTTP (requests) runs in a thread so scrapers overlap
                await asyncio.to_thread(self._run_cycle)
            except Exception as e:
                logger.error(f"[{self.name}] Error during scraping: {e}")
                safe_emit(emit_heartbeat, name="scraper", instance_id=self.name, status="FAIL")

            if self.fetch_interval <= 0:
                break
            await asyncio.sleep(self.fetch_interval)


async def main():
    parser = argparse.ArgumentParser(description="Continuous web scraper")
    parser.add_argument("--config", nargs="*", help="Specify which config(s) to use (none = all)")
    args = parser.parse_args()

    configs = load_configs(selected=args.config)
    scrapers = [Scraper(scraper_conf, mapping_conf) for scraper_conf, mapping_conf in configs]
    await asyncio.gather(*(s.run() for s in scrapers))


if __name__ == "__main__":
    asyncio.run(main())
