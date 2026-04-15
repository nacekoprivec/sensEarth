import os
import json
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import text
from psycopg2.extras import execute_values
from psycopg2.extras import Json


from ..database import get_db
from ..modeling_services.anomaly_detection_model import AnomalyDetectionModel
from ..modeling_services.forecast_model import ForecastModel

from .models import *
from .schemas import *
from .exceptions import *
from .utils import *
from ..logger import logger

from monitoring.client import emit_component_registration, emit_event, emit_metric, emit_heartbeat

from sqlalchemy.exc import IntegrityError

MODEL_REGISTRY = {
    "anomaly_detection_model": AnomalyDetectionModel,
    "forecast_model": ForecastModel,  
}

CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

def register_entities(payload: RegisterPayload, db: Session) -> Dict[str, Dict[str, int]]:
    """
    Registers nodes and sensors from the payload.
    Returns:
    {
        "nodes": {node_hash1: node_id1, ...},
        "sensors": {sensor_hash1: sensor_id1, ...}
    }
    """

    emit_heartbeat(name="middleware", instance_id="default", status="OK")
    db_healthcheck(db) 

    emit_event(name="middleware",
        instance_id="default",
        event_type="registering_entities",
        severity="INFO",
        message="Register endpoint called",
        metadata={"payload_size": len(payload.nodes) + len(payload.sensors)},
    )
    logger.info("Register endpoint called", extra={"nodes": len(payload.nodes), "sensors": len(payload.sensors)})

    node_map = {}          
    sensor_map = {}        

    # =========================
    # REGISTER NODES
    # =========================
    for node_data in payload.nodes:
        node_label = node_data.node_label
        node_hash = node_data.node_hash
        longitude = node_data.longitude
        latitude = node_data.latitude
        altitude = node_data.altitude
        node_description = node_data.description

        loc_sql, loc_params = create_location_params(longitude, latitude, altitude)

        q_node = text(f"""
            INSERT INTO sensor_node (node_label, node_hash, location, description, last_seen)
            VALUES (:label, :hash, {loc_sql}, :desc, NOW())
            ON CONFLICT (node_hash)
            DO UPDATE SET
                last_seen = NOW()
            RETURNING node_id
        """)

        try:
            node_id = db.execute(
                q_node,
                {
                    "label": node_label,
                    "hash": node_hash,
                    "desc": node_description,
                    **loc_params
                }
            ).fetchone()[0]

        except IntegrityError as e:
            emit_event(
                name="middleware",
                instance_id="default",
                event_type="register_node_failed",
                severity="ERROR",
                message=str(e),
                metadata={"node_hash": node_data.node_hash}
            )

            raise HTTPException(
                status_code=409,
                detail=f"Node conflict: node_hash: '{node_hash}' already exists"
            )

        node_map[node_hash] = node_id

    # =========================
    # REGISTER SENSORS
    # =========================
    for sensor_data in payload.sensors:
        # Use dot notation for Pydantic attributes
        sensor_label = sensor_data.sensor_label
        sensor_hash = sensor_data.sensor_hash
        node_hash = sensor_data.node_hash
        sensor_description = sensor_data.sensor_description

        node_id = node_map.get(node_hash)
        if node_id is None:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown node_hash '{node_hash}' for sensor '{sensor_label}'"
            )

        # -------------------------
        # SENSOR LOCATION
        # -------------------------
        longitude = sensor_data.longitude
        latitude = sensor_data.latitude
        altitude = sensor_data.altitude

        loc_sql, loc_params = create_location_params(longitude, latitude, altitude)

        # -------------------------
        # SENSOR TYPE UPSERT
        # -------------------------
        stype = sensor_data.sensor_type 
        q_type = text("""
            INSERT INTO sensor_type (name, phenomenon, unit, value_min, value_max)
            VALUES (:name, :phenomenon, :unit, :min, :max)
            ON CONFLICT (name)
            DO UPDATE SET
                name = EXCLUDED.name
            RETURNING sensor_type_id
        """)

        sensor_type_id = db.execute(
            q_type,
            {
                "name": stype.name,
                "phenomenon": stype.phenomenon,
                "unit": stype.unit,
                "min": stype.value_min,
                "max": stype.value_max,
            }
        ).fetchone()[0]

        # -------------------------
        # SENSOR INSERT
        # -------------------------
        q_sensor = text(f"""
            INSERT INTO sensor (node_id, sensor_type_id, sensor_hash, sensor_label, location, description, last_seen)
            VALUES (:node_id, :stype_id, :hash, :label, {loc_sql}, :desc, NOW())
            ON CONFLICT (node_id, sensor_hash)
            DO UPDATE SET
                last_seen = NOW(),
                sensor_label = EXCLUDED.sensor_label
            RETURNING sensor_id
        """)

        try:
            sensor_id = db.execute(
                q_sensor,
                {
                    "node_id": node_id,
                    "stype_id": sensor_type_id,
                    "hash": sensor_hash,
                    "label": sensor_label,
                    "desc": sensor_description,
                    **loc_params
                }
            ).fetchone()[0]

        except IntegrityError as e:
            emit_event(
                name="middleware",
                instance_id="default",
                event_type="register_sensor_failed",
                severity="ERROR",
                message=str(e),
                metadata={"sensor_hash": sensor_data.sensor_hash}
            )

            raise HTTPException(
                status_code=409,
                detail=f"Sensor conflict: sensor_hash '{sensor_hash}' already exists"
            )

        sensor_map[sensor_hash] = sensor_id

    emit_metric(name="middleware", instance_id="default", metric_name="registered_nodes", value=len(node_map), unit="count")
    emit_metric(name="middleware", instance_id="default", metric_name="registered_sensors", value=len(sensor_map), unit="count")

    db.commit()
    return {"nodes": node_map, "sensors": sensor_map}
        
def ingest_measurements(payload: dataIngestPayload, db: Session) -> Dict[str, Any]:
    """
    Ingests measurement data for sensors. Each entry in the payload must include 'sensor_hash', 'timestamp_utc', and 'value'.
    """
    emit_heartbeat(name="middleware", instance_id="default", status="OK")
    db_healthcheck(db) 

    measurement_buffer = []

    sensor_hashes = [
        measurement.sensor_hash
        for measurement in payload
    ]
    
    if None in sensor_hashes:
        raise HTTPException(status_code=400, detail="All entries must include 'sensor_hash'")

    if sensor_hashes:
        rows = db.execute(
            text("SELECT sensor_hash, sensor_id FROM sensor WHERE sensor_hash = ANY(:hashes)"),
            {"hashes": sensor_hashes}
        ).fetchall()
        hash_to_id = {row[0]: row[1] for row in rows}

    for measurement in payload:
        sensor_hash = measurement.sensor_hash
        ts = measurement.timestamp_utc
        value = measurement.value

        sensor_id_db = hash_to_id.get(sensor_hash)
        if sensor_id_db is None: # ID in database not found
            raise HTTPException(status_code=404, detail=f"Sensor '{sensor_hash}' not found")
        try:
            val = float(value)
            measurement_buffer.append((sensor_id_db, ts, val))
        except (TypeError, ValueError):
            logger.warning(f"Skipping invalid measurement: {measurement}")
            continue

    if measurement_buffer:  
        conn = db.get_bind().raw_connection()  # get psycopg2 connection
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    """
                    INSERT INTO sensor_measurement (sensor_id, timestamp_utc, value)
                    VALUES %s
                    ON CONFLICT (sensor_id, timestamp_utc)
                    DO UPDATE SET value = EXCLUDED.value
                    """,
                    measurement_buffer
                )
            conn.commit()
        finally:
            conn.close()
    db.commit()

    # TODO: check lenght of buffer
    emit_metric(
        name="middleware",
        instance_id="default",
        metric_name="measurements_ingested",
        value=len(measurement_buffer) if measurement_buffer else 0,
        unit="count"
    )

    emit_event(
        name="middleware",
        instance_id="default",
        event_type="data_ingest_completed",
        severity="INFO",
        message=f"Inserted {len(measurement_buffer)} measurements",
    )

    return {"status": "ok", "inserted_measurements": len(measurement_buffer)}

def create_model(payload: CreateModelPayload, db: Session):
    """
    Create a new model and optionally associate it with one or more sensors.

    Expected keys in payload:
    - model_name: str (required)
    - model_description: str (optional)
    - model_parameters: dict (optional, stored as JSONB)
    - sensor_id_list: list[int] (optional) Allows linking model to multiple sensors.

    If a sensor association fails (e.g., sensor ID not found), the model is still created but not linked to that sensor.

    Returns:
    {
        "model_id": int
    }
    """

    emit_heartbeat(name="database", instance_id="default", status="OK")
    db_healthcheck(db)

    emit_event(
        name="middleware",
        instance_id="default",
        event_type="registering_model",
        severity="INFO",
        message="Register model endpoint called",
        metadata={"model_name": payload.model_name}  
    )

    model_name = payload.model_name
    model_description = payload.model_description
    model_parameters = payload.model_parameters
    sensor_id_list = payload.sensor_id_list
    model_type = payload.model_type

    if not model_name or model_name == "":
        logger.warning("Model name is required")
        raise HTTPException(status_code=400, detail="model_name is required")

    try:
        # Create model
        q_model = text("""
            INSERT INTO model (name, description, model_type, parameters)
            VALUES (:name, :description, :model_type, :parameters)
            RETURNING model_id
        """)

        row = db.execute(
            q_model,
            {
                "name": model_name,
                "description": model_description,
                "model_type": model_type,
                "parameters": Json(model_parameters)
            }
        ).mappings().fetchone()

        if not row:
            raise HTTPException(status_code=500, detail="Failed to create model")
        
        emit_component_registration(name="anomaly_detection_model", instance_id=model_name, component_type="model")

        model_dict = dict(row)
        logger.info(f"Model '{model_name}' created model {model_dict}")

        # Optionally associate sensor
        associated_sensors = []

        for sensor_id in set(sensor_id_list):
            if sensor_id is not None:
                q_link = text("""
                    INSERT INTO model_sensor (model_id, sensor_id)
                    SELECT :model_id, :sensor_id
                    WHERE EXISTS (
                        SELECT 1 FROM sensor WHERE sensor_id = :sensor_id
                    )
                    ON CONFLICT DO NOTHING
                """)

                row_sensor = db.execute(
                    q_link,
                    {"model_id": model_dict["model_id"], "sensor_id": sensor_id}
                )

                associated_sensors.append(row_sensor)

                if row_sensor.rowcount == 0:
                    logger.warning(
                        f"Sensor ID {sensor_id} not found. Model created without association."
                    )
                    
        db.commit()
        model_dict["associated_sensors"] = len(associated_sensors)

        emit_event(
            name="middleware",
            instance_id="default",
            event_type="model_registration_success",
            severity="INFO",
            message=f"Model '{model_name}' registered successfully",
            metadata={"model_id": model_dict["model_id"], "associated_sensors": len(associated_sensors)}
        )

        emit_metric(
            name="middleware",
            instance_id="default",
            metric_name="registered_models",
            value=1,
            unit="count"
        )

        return model_dict
    except Exception as e:
        emit_event(
            name="middleware",
            instance_id="default",
            event_type="model_registration_failed",
            severity="ERROR",
            message=f"Failed to register model '{model_name}'",
            metadata={"error": str(e)}
        )

        db.rollback()
        logger.error("Error creating model", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create model")


async def model_results(payload: Dict, MODEL_REGISTRY: Dict[str, Any], db: Session) -> Dict[str, Any]:
    """
    Run specified model for a sensor with its configuration.
    Model name is UNIQUE so we can identify models by name.
    """

    emit_heartbeat(name="middleware", instance_id="default", status="OK")
    db_healthcheck(db)

    started_at = datetime.datetime.now(datetime.timezone.utc)

    emit_event(
        name="middleware",
        instance_id="default",
        event_type="model_run_started",
        severity="INFO",
        message="Run model endpoint called",
        metadata={"model_name": payload.get("model_name")}
    )

    model_name = payload.get("model_name")
    sensor_id_list = set(payload.get("sensor_id_list", []))
    sliding_window_size = payload.get("sliding_window_size", 100)

    if not model_name:
        logger.error("Missing model_name in runModel payload")
        raise HTTPException(status_code=400, detail="Missing required fields in payload")

    row_model = db.execute(
        text("SELECT model_id, model_type FROM model WHERE name = :model_name"),
        {"model_name": model_name}
    ).fetchone()

    if not row_model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Check the model name")

    model_id = row_model[0]
    model_type = row_model[1]

    row_params = db.execute(
        text("SELECT parameters FROM model WHERE model_id = :model_id"),
        {"model_id": model_id}
    ).fetchone()

    parameters = json.loads(row_params[0]) if isinstance(row_params[0], str) else row_params[0]
    logger.info(f"Using stored parameters for model '{model_name}': {parameters}")

    ModelClass = MODEL_REGISTRY.get(model_type) 
    if not ModelClass:
        raise HTTPException(status_code=400, detail=f"Unsupported model type '{model_type}'")

    logger.info(f"Model ID for '{model_name}': {model_id}, Model Type: {model_type}")

    if not sensor_id_list:
        sensor_ids = db.execute(
            text("SELECT sensor_id FROM model_sensor WHERE model_id = :model_id"),
            {"model_id": model_id}
        ).fetchall()

        sensor_id_list = set([row[0] for row in sensor_ids])

    logger.info(f"Sensor IDs associated with model '{model_name}': {sensor_id_list}")

    results = {}
    for sensor_id in sensor_id_list:
        row_sensor = db.execute(
            text("SELECT sensor_id FROM sensor WHERE sensor_id = :sensor_id"),
            {"sensor_id": sensor_id}
        ).fetchone()

        if not row_sensor:
            raise HTTPException(status_code=404, detail=f"Sensor '{sensor_id}' not found")

        sensor_id = row_sensor[0]

        model_instance = ModelClass(
            model_name=model_name,
            sensor_id=sensor_id,
            conf=parameters
        )

        model_instance.data = []
        sensor_data = db.execute(
            text("""
                SELECT timestamp_utc, value FROM sensor_measurement
                WHERE sensor_id = :sensor_id
                ORDER BY timestamp_utc ASC
                LIMIT :limit
            """),
            {"sensor_id": sensor_id, "limit": sliding_window_size}
        ).fetchall()

        model_instance.data_ingestion(sensor_data)

        results[sensor_id] = model_instance.run() # Return ex. ([20510.8125, 174.0], ('Error: measurement above upper limit', -1)) , ....
        logger.info(f"Model run completed for sensor_id {sensor_id}")
    
    finished_at = datetime.datetime.now(datetime.timezone.utc)

    model_run_row = db.execute(
        text(f"""
            INSERT INTO model_run (model_id, started_at, finished_at, status)
            VALUES (:model_id, :started_at, :finished_at, :status)
            RETURNING run_id
        """),
        {"model_id": model_id, "started_at": started_at, "finished_at": finished_at, "status": "completed"}
    ).fetchone()

    model_run_id = model_run_row[0]
    logger.info(f"Created model run with ID {model_run_id} for model '{model_name}'")
    
    unix_epoch = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
    rows = []
    
    for sensor_id, sensor_results in results.items():
        for entry in sensor_results:

            timestamp_float = entry[0][0]
            value = entry[0][1]

            if model_type == "anomaly_detection_model": # Writes the anomaly detection message
                inference_message = entry[1][0]

            elif model_type == "forecast_model": # Writes forecast prediction
                inference_message = entry[1][2]

            timestamp_utc = unix_epoch + timedelta(days=timestamp_float)

            rows.append({
                "run_id": model_run_id,
                "model_id": model_id,
                "sensor_id": sensor_id,
                "timestamp_utc": timestamp_utc,
                "value": value,
                "inference_message": inference_message
            })

        db.execute(
            text("""
                INSERT INTO model_inference
                (run_id, model_id, sensor_id, timestamp_utc, value, inference_message)
                VALUES
                (:run_id, :model_id, :sensor_id, :timestamp_utc, :value, :inference_message)
            """),
            rows
        )
        db.commit()

    emit_event(
        name="middleware",
        instance_id="default",
        event_type="model_run_completed",
        severity="INFO",
        message="Model run completed for all sensors",
        metadata={"model_name": payload.get("model_name"), "sensor_count": len(sensor_id_list)}
    )

    return {"status": "model run completed", "results": results}

def get_models(db: Session):
    """
    Fetch all registered models with their details.
    """

    db_healthcheck(db)

    rows = db.execute(
        text("SELECT model_id, name, description, model_type FROM model")
    ).mappings().fetchall()

    if not rows:
        logger.info("No models found in database")

    return rows
    
def get_model(model_name: str, db: Session):
    """
    Fetch specific model information by name.
    """
    db_healthcheck(db)

    try:
        row = db.execute(
            text("SELECT model_id, name, description, model_type, parameters FROM model WHERE name = :model_name"),
            {"model_name": model_name}
        ).mappings().fetchone()

        if not row:
            logger.warning(f"Model '{model_name}' not found in database")
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    except Exception as e:
        logger.error("Error fetching model", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch model")

    return row

def get_model_runs(db: Session):
    """
    Fetch all registered model runs with their details.
    """
    db_healthcheck(db)

    try:
        rows = db.execute(
            text("SELECT run_id, model_id, started_at, finished_at, status FROM model_run")
        ).mappings().fetchall()

        if not rows:
            logger.info("No model runs found in database")
            return []

        return rows
    except Exception as e:
        logger.error("Error fetching model runs", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch model runs")

def get_model_run_log(run_id: int, db: Session):
    """
    Fetch the logs for a specific model run.
    Fetch all logs if run_id is not provided.
    """
    db_healthcheck(db)
    try:
        if run_id:
            rows = db.execute(
                text("SELECT * FROM model_inference WHERE run_id = :run_id"),
                {"run_id": run_id}
            ).mappings().fetchall()
        else:
            rows = db.execute(
                text("SELECT * FROM model_inference")
            ).mappings().fetchall()

        if not rows:
            logger.info("No model run logs found in database")
            return []
        return rows
    except Exception as e:
        logger.error("Error fetching model run log", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch model run log")

def delete_models(db: Session) -> None:
    """
    Deletes all models from the database. This will delete all models and associated configurations and results.
    """
    db_healthcheck(db)

    try: 
        q_delete_all = text("DELETE FROM model")
        db.execute(q_delete_all)
        db.commit()
        logger.info("All models removed successfully")
    except Exception as e:
        db.rollback()
        logger.error("Error removing all models", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to remove all models")
    
    return {"status": "ok", "message": "All models removed successfully"}

def delete_model(model_name: str, db: Session):
    """
    Deletes specific model. This will delete the model and all associated configurations and results.
    """
    db_healthcheck(db)

    try:
        q_delete = text("""
            DELETE FROM model
            WHERE name = :model_name
            RETURNING model_id
        """)
        result = db.execute(q_delete, {"model_name": model_name})
        deleted_model_id = result.scalar_one_or_none()

        if deleted_model_id is None:
            raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

        db.commit()
        logger.info(f"Model '{model_name}' with id {deleted_model_id} removed successfully")
        return {"status": "ok", "message": f"Model '{model_name}' removed successfully"}
    except Exception as e:
        db.rollback()
        logger.error("Error removing model", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to remove model")

def get_nodes(db: Session):
    """
    Fetch all registered nodes with their details.
    """
    db_healthcheck(db)

    rows = db.execute(
        text("SELECT node_id, node_label, node_hash, description FROM sensor_node")
    ).mappings().fetchall()

    if not rows:
        logger.info("No nodes found in database")

    return rows

def get_node(node_id: int, db: Session):
    """
    Fetch specific node information by ID.
    """
    db_healthcheck(db)

    try:
        row = db.execute(
            text("SELECT node_id, node_label, node_hash, description FROM sensor_node WHERE node_id = :node_id"),
            {"node_id": node_id}
        ).mappings().fetchone()

        if not row:
            logger.warning(f"Node with ID '{node_id}' not found in database")
            raise HTTPException(status_code=404, detail=f"Node with ID '{node_id}' not found")
    except Exception as e:
        logger.error("Error fetching node", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch node")

    return row

def get_sensors(db: Session):
    """
    Fetch all registered sensors with their details.
    """
    db_healthcheck(db)

    rows = db.execute(
        text("""
            SELECT s.sensor_id, s.sensor_label, ST_AsGeoJSON(s.location) AS location, st.name, s.status AS sensor_status
            FROM sensor s
            JOIN sensor_type st ON s.sensor_type_id = st.sensor_type_id
        """)
    ).mappings().fetchall()

    if not rows:
        logger.info("No sensors found in database")

    return rows

def get_sensor(sensor_id: int, db: Session):
    """
    Fetch specific sensor information by ID.
    """
    db_healthcheck(db)

    try:
        row = db.execute(
            text("""
                SELECT s.sensor_id, s.sensor_label, s.sensor_hash, s.description, st.name AS sensor_type
                FROM sensor s
                JOIN sensor_type st ON s.sensor_type_id = st.sensor_type_id
                WHERE s.sensor_id = :sensor_id
            """),
            {"sensor_id": sensor_id}
        ).mappings().fetchone()

        if not row:
            logger.warning(f"Sensor with ID '{sensor_id}' not found in database")
            raise HTTPException(status_code=404, detail=f"Sensor with ID '{sensor_id}' not found")
    except Exception as e:
        logger.error("Error fetching sensor", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch sensor")

    return row

def get_measurements(sensorIDs: List[int], days: int, limit: int, db: Session):
    """
    Fetch measurements for specific sensors from the database.
    """
    db_healthcheck(db)

    try:
        if days > 0 and sensorIDs: # Measurements from last N days for specific sensors

            time_threshold = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=days)
            rows = db.execute(
                text("""
                    SELECT sm.timestamp_utc, sm.value, s.sensor_id, s.sensor_label, ST_AsGeoJSON(s.location) AS location
                    FROM sensor_measurement sm 
                    INNER JOIN sensor s ON sm.sensor_id = s.sensor_id
                    WHERE sm.sensor_id = ANY(:sensor_ids) AND sm.timestamp_utc >= :time_threshold
                    ORDER BY sm.timestamp_utc DESC
                """),
                {"sensor_ids": sensorIDs, "time_threshold": time_threshold, "limit": limit}
            ).mappings().fetchall()
        
        else:
            # Return latest N measurements without time filtering
            rows = db.execute(
                text("""
                    SELECT sm.timestamp_utc, sm.value, s.sensor_id, s.sensor_label, ST_AsGeoJSON(s.location) AS location
                    FROM sensor_measurement sm INNER JOIN sensor s ON sm.sensor_id = s.sensor_id
                    ORDER BY sm.timestamp_utc DESC
                    LIMIT :limit
                """),
                {"limit": limit}
            ).mappings().fetchall()

        if not rows:
            logger.info(f"No measurements found")
            return []
    except Exception as e:
        logger.error("Error fetching sensor measurements", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to fetch sensor measurements")

    return rows