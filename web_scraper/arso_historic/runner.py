"""
Orchestrate ARSO historic daily import for known sensors.

Pipeline: GET /sensors (sifra) -> fetch daily CSV -> parse/convert/guard
-> dry-run report or chunked POST /dataIngest. Never registers sensors.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.exceptions import RequestException

from logger import logger
from utils import request_with_retry

from arso_historic.daily_csv import parse_daily_csv
from arso_historic.fetch import build_daily_csv_url, fetch_daily_csv
from arso_historic.guards import filter_known_measurements, known_sensor_hashes
from arso_historic.sensors import fetch_sensors, group_sensors_by_sifra, sifra_of


def default_api_url() -> str:
    # MIDDLEWARE_API or local default; strip trailing slash once so callers
    # can join paths without re-stripping.
    return os.getenv("MIDDLEWARE_API", "http://localhost:5006").rstrip("/")


def assert_source_column_available(api_url: str) -> None:
    """
    Hard-fail before historic ingest when the provenance column is missing.

    Fix 5: historic rows must stay deletable by source tag, so refuse to
    write until the migration has been applied.
    """
    response = request_with_retry(
        requests.get,
        f"{api_url}/measurements/provenance",
        timeout=30,
    )
    body = response.json()

    # Fix 4: be explicit about an unexpected response shape
    if not isinstance(body, dict) or "source_column_available" not in body:
        raise RuntimeError(
            f"Unexpected /measurements/provenance response: {body!r}"
        )
    if not body["source_column_available"]:
        raise RuntimeError(
            "sensor_measurement.source column is not available. "
            "Run database/migrations/add_measurement_source.sql, restart the API, "
            "then retry --ingest. Dry-run remains allowed."
        )


def process_station(
    sifra: str,
    sensors_by_type: Dict[str, Dict[str, Any]],
    year_from: int,
    year_to: int,
    source_tag: Optional[str] = None,
    include_measurements: bool = True,
) -> Dict[str, Any]:
    """
    Fetch + parse one station. Does not POST.

    include_measurements=False keeps only a small sample (Fix 7/23): dry-run
    does not need the full payload retained in memory.
    """
    # Provenance tag consumed by DELETE /measurements/by-source (Fix 9)
    source = source_tag or f"historic:arso:{sifra}"

    url = build_daily_csv_url(sifra, year_from, year_to)
    logger.info(f"[arso_historic] fetching sifra={sifra} {year_from}-{year_to}")
    logger.info(f"[arso_historic] url={url}")

    # 1) download  2) parse/convert  3) defense-in-depth hash filter (Fix 11)
    raw = fetch_daily_csv(sifra, year_from, year_to)
    parsed = parse_daily_csv(raw, sensors_by_type)
    known = known_sensor_hashes(sensors_by_type)
    guarded = filter_known_measurements(parsed["measurements"], known)

    # 4) shape ingest payload. value is stringified because the API
    # MeasurementPayload.value is Optional[str] (Fix 8).
    sample: List[Dict[str, Any]] = []
    measurements: List[Dict[str, Any]] = []
    ready = 0
    for item in guarded["measurements"]:
        entry = {
            "sensor_hash": item["sensor_hash"],
            "timestamp_utc": item["timestamp_utc"],
            "value": str(item["value"]),
            "source": source,
        }
        ready += 1
        if len(sample) < 5:
            sample.append(entry)
        if include_measurements:
            measurements.append(entry)

    return {
        "sifra": sifra,
        "url": url,
        "sensors": {
            st: {
                "sensor_id": s.get("sensor_id"),
                "sensor_hash": s.get("sensor_hash"),
                "unit": s.get("unit"),
                "sensor_label": s.get("sensor_label"),
            }
            for st, s in sensors_by_type.items()
        },
        "headers": parsed["headers"],
        "column_map": parsed["column_map"],
        "conversions": parsed["conversions"],
        "warnings": parsed["warnings"],
        "stats": {
            **parsed["stats"],
            "skipped_unknown_hash": guarded["skipped_unknown_hash"],
            "ready_to_ingest": ready,
        },
        "sample": sample,
        "measurements": measurements,
        "source": source,
    }


def send_measurement_chunks(
    api_url: str,
    measurements: List[Dict[str, Any]],
    chunk_size: int = 500,
    chunk_delay_s: float = 0.0,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    POST /dataIngest in chunks to avoid one huge payload (Fix 18).

    Returns (api_bodies, inserted_total). inserted_total sums the API's
    reported inserted_measurements (Fix 21) rather than assuming every row
    was written. Partial success is possible on mid-run failure; roll back
    with DELETE /measurements/by-source (Fix 14).
    """
    # Fix 15: guard invalid chunk size for direct run() callers (CLI also checks)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")

    results: List[Dict[str, Any]] = []
    inserted_total = 0
    for i in range(0, len(measurements), chunk_size):
        chunk = measurements[i : i + chunk_size]
        response = request_with_retry(
            requests.post,
            f"{api_url}/dataIngest",
            json=chunk,
            timeout=60,
        )
        body = response.json()
        results.append(body)

        # Trust the API's own count when present
        if isinstance(body, dict):
            inserted_total += int(body.get("inserted_measurements", 0) or 0)

        logger.info(
            f"[arso_historic] ingested {i + len(chunk)}/{len(measurements)} → {body}"
        )

        # Fix 16: optional throttle between chunks for bulk runs
        if chunk_delay_s > 0:
            time.sleep(chunk_delay_s)

    return results, inserted_total


def run(
    *,
    sifra: Optional[str] = None,
    year_from: int = 1900,
    year_to: Optional[int] = None,
    dry_run: bool = True,
    active_only: bool = True,
    api_url: Optional[str] = None,
    chunk_size: int = 500,
    station_delay_s: float = 0.0,
) -> Dict[str, Any]:
    """
    CLI entry point.

    sifra=None processes every grouped station (CLI requires --all-stations).
    The provenance gate runs only when dry_run=False (Fix 24/26).
    """
    api = api_url or default_api_url()
    if year_to is None:
        year_to = datetime.now(timezone.utc).year

    if not dry_run:
        assert_source_column_available(api)

    sensors = fetch_sensors(api, active_only=active_only)

    # Fix 19: filter to the requested station BEFORE grouping so an ambiguous
    # (conflicting) sensor at some *other* sifra cannot abort a single-station run.
    if sifra:
        sifra = str(sifra).strip()
        available = sorted({s for s in (sifra_of(x) for x in sensors) if s})
        sensors = [s for s in sensors if sifra_of(s) == sifra]
        if not sensors:
            raise ValueError(
                f"No known sensors with sifra={sifra}. "
                f"Stations with sifra in DB: {available[:20]}"
                + ("..." if len(available) > 20 else "")
            )

    # Single-station runs are strict (loud on ambiguity); bulk runs skip
    # conflicted stations and keep going (Fix 8 / sensors.strict).
    grouped = group_sensors_by_sifra(sensors, strict=bool(sifra))
    if sifra:
        if sifra not in grouped:
            raise ValueError(
                f"sifra={sifra} has no unambiguous hydro sensor to import"
            )
        stations = {sifra: grouped[sifra]}
    else:
        stations = grouped

    logger.info(
        f"[arso_historic] stations={len(stations)} "
        f"years={year_from}-{year_to} dry_run={dry_run} api={api}"
    )

    reports = []
    totals = {"ready": 0, "ingested": 0, "failed_stations": 0}

    for station_id, by_type in sorted(stations.items()):
        try:
            report = process_station(
                station_id,
                by_type,
                year_from,
                year_to,
                include_measurements=not dry_run,  # Fix 7/23
            )
            reports.append({k: v for k, v in report.items() if k != "measurements"})
            totals["ready"] += report["stats"]["ready_to_ingest"]

            for warning in report["warnings"]:
                logger.warning(f"[arso_historic] sifra={station_id}: {warning}")

            logger.info(
                f"[arso_historic] sifra={station_id} "
                f"kept={report['stats']['kept']} "
                f"ready={report['stats']['ready_to_ingest']} "
                f"conversions={report['conversions']}"
            )

            if not dry_run and report["measurements"]:
                _, inserted = send_measurement_chunks(
                    api, report["measurements"], chunk_size=chunk_size
                )
                totals["ingested"] += inserted  # Fix 21: API-reported count

        # Fix 20: known failures fail just this station; unexpected errors are
        # logged with traceback (still isolated, not silently swallowed).
        except (RequestException, ValueError, RuntimeError, KeyError) as exc:
            totals["failed_stations"] += 1
            logger.error(f"[arso_historic] sifra={station_id} failed: {exc}")
            reports.append({"sifra": station_id, "error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - isolate unknown per-station errors
            totals["failed_stations"] += 1
            logger.error(
                f"[arso_historic] sifra={station_id} unexpected failure",
                exc_info=True,
            )
            reports.append({"sifra": station_id, "error": str(exc)})

        # Fix 22: optional throttle between stations for --all-stations runs
        if station_delay_s > 0:
            time.sleep(station_delay_s)

    return {
        "dry_run": dry_run,
        "year_from": year_from,
        "year_to": year_to,
        "station_count": len(stations),
        "totals": totals,
        "reports": reports,
    }
