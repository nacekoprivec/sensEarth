"""
Resolve known ARSO hydro sensors from middleware GET /sensors.

Filters sensors that have metadata.sifra / node_serial and groups by station.
Callers rely on each sensor dict exposing sensor_hash, unit, sensor_type,
and value_min/value_max (from GET /sensors).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import requests

from logger import logger
from utils import request_with_retry
from enricher import normalize_sifra


HYDRO_SENSOR_TYPES = frozenset(
    {
        "water_level",
        "water_flow",
        "water_temperature",
    }
)


def fetch_sensors(api_url: str, active_only: bool = True) -> List[Dict[str, Any]]:
    """
    GET /sensors or /sensors/active and return the JSON list.

    active_only defaults True; historic-only stations are often inactive, so
    the CLI exposes --all-statuses to include them (Fix 2).
    """
    path = "/sensors/active" if active_only else "/sensors"
    # Fix 1/3: shared helper retries 5xx and connection errors
    response = request_with_retry(
        requests.get,
        f"{api_url.rstrip('/')}{path}",
        timeout=30,
    )
    try:
        data = response.json()
    except ValueError as exc:
        raise ValueError(f"/sensors returned non-JSON response: {exc}") from exc

    if not isinstance(data, list):
        raise ValueError(f"Unexpected /sensors response type: {type(data)}")
    return data


def sifra_of(sensor: Dict[str, Any]) -> Optional[str]:
    """
    Return the ARSO station id for a sensor, or None.

    node_serial is the API alias of metadata.sifra; prefer it, then fall back
    to raw metadata. normalize_sifra strips blanks and collapses integer-like
    values (e.g. "9275.0" -> "9275") (Fix 5/6/7).
    """
    metadata = sensor.get("metadata")
    meta_sifra = metadata.get("sifra") if isinstance(metadata, dict) else None

    for candidate in (sensor.get("node_serial"), meta_sifra):
        cleaned = normalize_sifra(candidate)
        if cleaned:
            return cleaned
    return None


def group_sensors_by_sifra(
    sensors: List[Dict[str, Any]],
    sensor_types: Optional[frozenset] = None,
    strict: bool = True,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Group hydro sensors that have a sifra by station id and sensor_type.

    Returns { sifra: { sensor_type: sensor_dict } }.

    A conflict is two DIFFERENT sensor_hash values for the same (sifra,
    sensor_type) — an ambiguous historic mapping. That (sifra, type) is
    dropped entirely (we cannot pick a target deterministically, Fix 9/10).

    strict=True  -> raise ValueError on any conflict (single-station runs).
    strict=False -> skip conflicted (sifra, type), log, and continue (bulk,
                    Fix 8).
    """
    allowed = sensor_types if sensor_types is not None else HYDRO_SENSOR_TYPES
    by_sifra: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
    conflicted: set[Tuple[str, str]] = set()
    conflicts: List[str] = []

    for sensor in sensors:
        sifra = sifra_of(sensor)
        if not sifra:
            continue

        # sensor_type is preferred; name is the legacy field
        sensor_type = sensor.get("sensor_type") or sensor.get("name")
        if sensor_type not in allowed:
            continue

        existing = by_sifra[sifra].get(sensor_type)
        if existing is not None and existing.get("sensor_hash") != sensor.get("sensor_hash"):
            conflicts.append(
                f"sifra={sifra} type={sensor_type}: "
                f"{existing.get('sensor_id')} vs {sensor.get('sensor_id')}"
            )
            conflicted.add((sifra, sensor_type))
            continue

        by_sifra[sifra][sensor_type] = sensor

    # Drop ambiguous (sifra, type) pairs; remove any station left empty
    for sifra, sensor_type in conflicted:
        by_sifra.get(sifra, {}).pop(sensor_type, None)
    grouped = {sifra: types for sifra, types in by_sifra.items() if types}

    if conflicts:
        message = (
            "Ambiguous sensors (multiple sensor_hash for one sifra+type): "
            + "; ".join(conflicts)
        )
        if strict:
            raise ValueError(message)
        logger.warning(f"[arso_historic] {message}")

    return grouped
