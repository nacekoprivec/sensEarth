from logger import logger
from sqlalchemy.orm import Session
from sqlalchemy import text
from psycopg2.extras import Json 
import traceback
from utils import *

component_id_cache: dict[tuple[str, str], int] = {}

def get_component_id(name: str, instance_id: str, db: Session) -> int | None:
    """
    Resolve component_id using cache + DB.
    Component must already be registered.
    """
    cache_key = (name, instance_id)

    # Check cache
    if cache_key in component_id_cache:
        return component_id_cache[cache_key]

    # DB lookup
    sql = text("""
        SELECT component_id
        FROM components
        WHERE name = :name AND instance_id = :instance_id
    """)
    result = db.execute(sql, {"name": name, "instance_id": instance_id})
    row = result.fetchone()

    if row is None:
        return None

    component_id = row[0]

    # Populate cache
    component_id_cache[cache_key] = component_id
    return component_id    

def save_component(payload: dict, db: Session):
    if "name" not in payload or "instance_id" not in payload:
        raise ValueError("name and instance_id are required")

    sql = text("""
        INSERT INTO components (name, instance_id, type, status)
        VALUES (:name, :instance_id, :type, :status)
        ON CONFLICT (name, instance_id) DO NOTHING
        RETURNING component_id
    """)

    result = db.execute(sql, {
        "name": payload["name"],
        "instance_id": payload["instance_id"],
        "type": payload.get("type", "other"),
        "status": payload.get("status", "active"),
    })
    db.commit()

    component_id = result.scalar()

    # If it already existed, fetch it
    if component_id is None:
        component_id = get_component_id(payload["name"], payload["instance_id"], db)

    # Seed cache
    component_id_cache[(payload["name"], payload["instance_id"])] = component_id

    return {"status": "component registered", "component_id": component_id}

def save_event(payload: dict, db: Session):
    component_id = get_component_id(payload["name"], payload["instance_id"], db)

    if component_id is None:
        raise ValueError(
            f"Component {payload['name']}:{payload['instance_id']} is not registered"
        )

    sql = text("""
        INSERT INTO events (component_id, event_type, severity, message, metadata)
        VALUES (:component_id, :event_type, :severity, :message, :metadata)
        RETURNING event_id
    """)

    result = db.execute(sql, {
        "component_id": component_id,
        "event_type": payload["event_type"],
        "severity": payload.get("severity", "INFO"),
        "message": payload.get("message"),
        "metadata": Json(payload.get("metadata") or {})
    })
    db.commit()

    return {"status": "event saved", "event_id": result.scalar()}


def save_metric(payload: dict, db: Session):
    component_id = get_component_id(payload["name"], payload["instance_id"], db)

    if component_id is None:
        raise ValueError(
            f"Component {payload['name']}:{payload['instance_id']} is not registered"
        )

    sql = text("""
        INSERT INTO metrics (component_id, metric_name, value, unit, metadata)
        VALUES (:component_id, :metric_name, :value, :unit, :metadata)
        RETURNING metric_id
    """)

    result = db.execute(sql, {
        "component_id": component_id,
        "metric_name": payload["metric_name"],
        "value": payload["value"],
        "unit": payload.get("unit"),    
        "metadata": Json(payload.get("metadata") or {})
    })
    db.commit()

    return {"status": "metric saved", "metric_id": result.scalar()}

def save_heartbeat(payload: dict, db: Session):
    component_id = get_component_id(payload["name"], payload["instance_id"], db)

    if component_id is None:
        raise ValueError(
            f"Component {payload['name']}:{payload['instance_id']} is not registered"
        )

    sql = text("""
        INSERT INTO heartbeats (component_id, status, metadata)
        VALUES (:component_id, :status, :metadata)
        RETURNING heartbeat_id
    """)

    result = db.execute(sql, {
        "component_id": component_id,
        "status": payload.get("status", "OK"),
        "metadata": Json(payload.get("metadata") or {})
    })
    db.commit()

    return {"status": "heartbeat saved", "heartbeat_id": result.scalar()}

def get_components_db(db: Session):
    try:
        rows = db.execute(text("SELECT * FROM components")).mappings().all()
        return rows
    except Exception as e:
        logger.error(f"Error fetching components: {e}")
        traceback.print_exc()
        return []

def get_events_db(db: Session):
    try:
        rows = db.execute(text("""
            SELECT e.*, c.name AS component_name, c.instance_id AS component_instance_id
            FROM events e
            JOIN components c ON e.component_id = c.component_id
            ORDER BY e.timestamp DESC
            LIMIT 100
        """)).mappings().all()
        logger.info(f"Fetched {len(rows)} events from database")
        return rows
    except Exception as e:
        logger.error(f"Error fetching events: {e}")
        traceback.print_exc()
        return []

def get_metrics_db(db: Session):
    try:
        rows = db.execute(text("""
            SELECT m.*, c.name AS component_name, c.instance_id AS component_instance_id
            FROM metrics m
            JOIN components c ON m.component_id = c.component_id
            ORDER BY m.timestamp DESC
            LIMIT 100
        """)).mappings().all()
        logger.info(f"Fetched {len(rows)} metrics from database")
        return rows
    except Exception as e:
        logger.error(f"Error fetching metrics: {e}")
        traceback.print_exc()
        return []

def delete_component_db(name: str, instance_id: str, db: Session):
    try:
        db.execute(text("DELETE FROM components WHERE name = :name AND instance_id = :instance_id"), {"name": name, "instance_id": instance_id})
        db.commit()
        component_id = component_id_cache.get((name, instance_id))
        keys_to_delete = [key for key, cid in component_id_cache.items() if cid == component_id]
        for key in keys_to_delete:
            del component_id_cache[key]
        logger.info(f"Deleted component {name}:{instance_id} and cleared {len(keys_to_delete)} cache entries")
        return {"status": "component deleted", "component_id": component_id}
    except Exception as e:
        logger.error(f"Error deleting component {component_id}: {e}")
        traceback.print_exc()
        return {"status": "error", "details": str(e)}