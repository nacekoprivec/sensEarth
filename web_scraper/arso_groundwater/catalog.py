"""
Load ARSO underground-water stations from Atlas okolja layer 23.

Layer 23 (lay_MM_POD_VODE) is the quantity-monitoring catalog: each feature
carries SIFRA_MM (archive p_postaja), IME_MM, LOKACIJA, and WGS84 geometry.
Layer 266 has names/coords only and is not used here.

This module only fetches and normalizes the catalog. It does not register
nodes/sensors or call the historic archive.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import requests

from utils import request_with_retry

# Query endpoint returns feature values; .../23?f=pjson is schema-only.
LAYER_23_QUERY_URL = (
    "https://gis.arso.gov.si/arcgis/rest/services/"
    "Atlasokolja_javni_D96/MapServer/23/query"
)

# ArcGIS may truncate large layers; layer 23 is small (~157) but guard anyway.
DEFAULT_QUERY_PARAMS = {
    "where": "1=1",
    "outFields": "*",
    "returnGeometry": "true",
    "outSR": "4326",
    "f": "geojson",
}

# WGS84 bounds for Slovenia with a small margin. Values outside usually mean
# outSR was ignored and coordinates are still in D96 metres.
SLOVENIA_LON_RANGE = (13.0, 17.0)
SLOVENIA_LAT_RANGE = (45.0, 47.5)

# Strip only a trailing "(cadastre code)" from IME_MM, not an earlier '('.
_TRAILING_PAREN = re.compile(r"\s*\([^)]*\)\s*$")


def validate_layer_23_geojson(body: Any, *, content_type: str = "", raw_preview: str = "") -> Dict[str, Any]:
    """
    Validate a layer-23 response body as a usable FeatureCollection.

    Separated from HTTP so tests can cover truncation / shape without network.
    """
    ctype = (content_type or "").lower()
    preview = raw_preview or ""
    if "json" not in ctype and preview.lstrip().startswith("<"):
        raise RuntimeError(
            "Layer 23 returned HTML instead of GeoJSON "
            f"(content-type={ctype!r}). "
            "GIS portal may be temporarily unavailable. "
            f"Preview: {preview!r}"
        )

    if not isinstance(body, dict):
        raise RuntimeError(f"Layer 23 response is not an object: {type(body)}")

    if body.get("type") != "FeatureCollection":
        raise RuntimeError(
            f"Layer 23 expected FeatureCollection, got type={body.get('type')!r}"
        )

    if "features" not in body or not isinstance(body["features"], list):
        raise RuntimeError("Layer 23 response missing features list")

    # ArcGIS sets this when maxRecordCount cuts the result short.
    if body.get("exceededTransferLimit"):
        raise RuntimeError(
            "Layer 23 query hit exceededTransferLimit; "
            "result is truncated and cannot be used as a full catalog"
        )

    return body


def fetch_layer_23_geojson(timeout: int = 60) -> Dict[str, Any]:
    """
    GET layer 23 as GeoJSON FeatureCollection.

    Raises RuntimeError when the body is HTML, not a FeatureCollection, has no
    features list, or reports exceededTransferLimit (silent truncation).
    """
    response = request_with_retry(
        requests.get,
        LAYER_23_QUERY_URL,
        params=DEFAULT_QUERY_PARAMS,
        timeout=timeout,
    )
    content_type = (response.headers.get("content-type") or "").lower()
    raw_preview = response.content[:200].decode("utf-8", errors="replace")

    # Detect HTML before response.json() so maintenance pages get a clear error.
    if "json" not in content_type and raw_preview.lstrip().startswith("<"):
        validate_layer_23_geojson(
            {}, content_type=content_type, raw_preview=raw_preview
        )

    try:
        body = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Layer 23 returned non-JSON response: {exc}. "
            f"content-type={content_type!r} preview={raw_preview!r}"
        ) from exc

    return validate_layer_23_geojson(
        body, content_type=content_type, raw_preview=raw_preview
    )


def _clean_text(value: Any) -> Optional[str]:
    """Strip whitespace; empty / non-string scalars become None."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return str(value).strip() or None
    text = str(value).strip()
    return text or None


def station_id_from_sifra(value: Any) -> Optional[str]:
    """
    Normalize SIFRA_MM for hashing and archive URLs.

    Only accepts strings. Numeric JSON values are rejected: leading zeros
    (e.g. '01094') cannot be recovered from int 1094, and a guessed pad would
    make hashes unstable. Schema types SIFRA_MM as string; ArcGIS should keep it.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def display_name_from_label(archive_label: str) -> str:
    """
    Short place name from IME_MM by removing a trailing '(code)' only.

    'Preserje (0430)' -> 'Preserje'
    'Foo (bar) (0430)' -> 'Foo (bar)'  # only the last parenthetical is stripped
    """
    name = _TRAILING_PAREN.sub("", archive_label).strip()
    return name or archive_label


def coords_in_slovenia(longitude: float, latitude: float) -> bool:
    lon_min, lon_max = SLOVENIA_LON_RANGE
    lat_min, lat_max = SLOVENIA_LAT_RANGE
    return lon_min <= longitude <= lon_max and lat_min <= latitude <= lat_max


def parse_station_feature(
    feature: Dict[str, Any],
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Map one GeoJSON feature to (station_dict, None) or (None, skip_reason).

    Required: string SIFRA_MM and Point geometry with WGS84 lon/lat in Slovenia.
    Optional: IME_MM, LOKACIJA, OBJECTID, VTPODV_ID (metadata only).
    """
    if not isinstance(feature, dict):
        return None, "feature is not an object"

    properties = feature.get("properties")
    if not isinstance(properties, dict):
        return None, "missing properties object"

    raw_sifra = properties.get("SIFRA_MM")
    if isinstance(raw_sifra, (int, float)) and not isinstance(raw_sifra, bool):
        return None, (
            f"SIFRA_MM is numeric ({raw_sifra!r}); "
            "leading zeros cannot be recovered — expected a string"
        )

    station_id = station_id_from_sifra(raw_sifra)
    if not station_id:
        return None, "missing or empty SIFRA_MM string"

    geometry = feature.get("geometry")
    if not isinstance(geometry, dict) or geometry.get("type") != "Point":
        return None, "geometry is not a Point"

    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, (list, tuple)) or len(coordinates) < 2:
        return None, "Point coordinates missing"

    try:
        longitude = float(coordinates[0])
        latitude = float(coordinates[1])
    except (TypeError, ValueError):
        return None, "Point coordinates are not numeric"

    if not coords_in_slovenia(longitude, latitude):
        return None, (
            f"coordinates outside Slovenia WGS84 bounds "
            f"(lon={longitude}, lat={latitude}); "
            f"expected lon in {SLOVENIA_LON_RANGE}, lat in {SLOVENIA_LAT_RANGE}"
        )

    archive_label = _clean_text(properties.get("IME_MM")) or station_id
    name = display_name_from_label(archive_label)

    station: Dict[str, Any] = {
        "station_id": station_id,
        "name": name,
        "archive_label": archive_label,
        "aquifer": _clean_text(properties.get("LOKACIJA")),
        "longitude": longitude,
        "latitude": latitude,
        "water_body_id": _clean_text(properties.get("VTPODV_ID")),
        "gis_objectid": properties.get("OBJECTID"),
    }
    return station, None


def stations_from_feature_collection(
    body: Dict[str, Any],
    *,
    source: str = LAYER_23_QUERY_URL,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse a validated FeatureCollection into stations + report.

    Raises RuntimeError on duplicate SIFRA_MM or when no usable stations remain.
    Safe to call with a fixture body (no network).
    """
    body = validate_layer_23_geojson(body)
    features = body["features"]

    stations: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    seen_ids: Dict[str, int] = {}
    duplicates: List[str] = []

    for index, feature in enumerate(features):
        parsed, reason = parse_station_feature(feature)
        props = feature.get("properties") if isinstance(feature, dict) else None

        if parsed is None:
            skipped.append(
                {
                    "index": index,
                    "objectid": props.get("OBJECTID") if isinstance(props, dict) else None,
                    "sifra_mm": props.get("SIFRA_MM") if isinstance(props, dict) else None,
                    "reason": reason or "unusable feature",
                }
            )
            continue

        station_id = parsed["station_id"]
        if station_id in seen_ids:
            duplicates.append(
                f"SIFRA_MM={station_id!r} at index {index} "
                f"(first at index {seen_ids[station_id]})"
            )
            continue

        seen_ids[station_id] = index
        stations.append(parsed)

    if duplicates:
        raise RuntimeError(
            "Duplicate SIFRA_MM in layer 23 catalog (ambiguous station id): "
            + "; ".join(duplicates)
        )

    stations.sort(key=lambda s: s["station_id"])

    if not stations:
        raise RuntimeError(
            "Layer 23 catalog produced zero usable stations "
            f"(features={len(features)}, skipped={len(skipped)})"
        )

    report = {
        "source": source,
        "feature_count": len(features),
        "station_count": len(stations),
        "skipped_count": len(skipped),
        "skipped": skipped,
    }
    return stations, report


def load_groundwater_stations(
    timeout: int = 60,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Fetch layer 23 and return (stations, report).

    stations: usable station dicts, sorted by station_id.
    report: counts and skip reasons for dry-run inspection.
    """
    body = fetch_layer_23_geojson(timeout=timeout)
    return stations_from_feature_collection(body, source=LAYER_23_QUERY_URL)
