"""
Guards for ARSO historic ingest candidates.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Set


def known_sensor_hashes(sensors_by_type: Dict[str, Dict[str, Any]]) -> Set[str]:
    """Set of sensor_hash for the station's registered sensors (skips any missing)."""
    return {
        s["sensor_hash"]
        for s in sensors_by_type.values()
        if s.get("sensor_hash")
    }


def filter_known_measurements(
    measurements: Iterable[Dict[str, Any]],
    known_hashes: Set[str],
) -> Dict[str, Any]:
    """
    Keep only measurements whose sensor_hash is in the known set.
    """
    kept: List[Dict[str, Any]] = []
    skipped = 0
    for item in measurements:
        sensor_hash = item.get("sensor_hash")
        if sensor_hash not in known_hashes:
            skipped += 1
            continue
        kept.append(item)
    return {"measurements": kept, "skipped_unknown_hash": skipped}
