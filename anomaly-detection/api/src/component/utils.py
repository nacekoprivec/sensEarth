from typing import Any, Dict, List, Optional, Tuple
from monitoring.client import emit_heartbeat
from sqlalchemy import text
from sqlalchemy.orm import Session


from typing import Any, Dict, List, Optional, Tuple
from monitoring.client import emit_heartbeat
from sqlalchemy import text
from sqlalchemy.orm import Session


def normalize_sifra(value: Any) -> Optional[str]:
    """
    Normalize ARSO station id (sifra) for metadata and archive URLs.

    - Integer-like values become digit strings (9275.0 -> "9275")
    - Blank / placeholder "sifra" -> None
    - Non-numeric codes (e.g. S-0759) pass through stripped
    """
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, int):
        return str(value)

    if isinstance(value, float):
        if not value.is_integer():
            return str(value).strip()
        return str(int(value))

    text_value = str(value).strip()
    if not text_value or text_value == "sifra":
        return None

    try:
        as_float = float(text_value)
        if as_float.is_integer():
            return str(int(as_float))
    except ValueError:
        pass

    return text_value


def prepare_sensor_metadata_for_upsert(
    raw_metadata: Optional[Dict[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """
    Prepare sensor metadata for register upsert.

    Merge-only by default: omitted or blank keys do not change stored metadata.
    Send an explicit JSON null to remove a key, e.g. {"sifra": null}.

    Returns:
        (merge_patch, remove_keys)
        - merge_patch is None when raw_metadata is None (do not touch metadata)
        - merge_patch may be {} when only removals were requested
    """
    if raw_metadata is None:
        return None, []

    merge: Dict[str, Any] = {}
    remove: List[str] = []

    for key, value in raw_metadata.items():
        if value is None:
            remove.append(key)
            continue

        if key == "sifra":
            sifra = normalize_sifra(value)
            if sifra is None:
                continue
            merge[key] = sifra
        else:
            merge[key] = value

    return merge, remove


def create_location_params(longitude: Optional[float], latitude: Optional[float], altitude: Optional[float] = None):
    """
    Returns a tuple of (SQL fragment, params dict) for inserting a PostGIS location.
    If longitude or latitude is missing, returns ("NULL", {}).
    """
    if longitude is not None and latitude is not None:
        if altitude is not None:
            return "ST_SetSRID(ST_MakePoint(:lon, :lat, :alt), 4326)", {"lon": longitude, "lat": latitude, "alt": altitude}
        return "ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)", {"lon": longitude, "lat": latitude}
    return "NULL", {}

def db_healthcheck(db: Session):
    try:
        db.execute(text("SELECT 1;"))
        emit_heartbeat(name="database", instance_id="default", status="OK")

        return {"status": "connected"}
    except Exception as e:
        emit_heartbeat(name="database", instance_id="default", status="FAIL")
        
        return {"status": "error", "details": str(e)}
