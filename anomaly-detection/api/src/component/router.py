import os
from typing import Dict, Any, List
from datetime import datetime

from importlib_metadata import metadata

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request, UploadFile, File, Form, Query
from fastapi.responses import JSONResponse

from sqlalchemy.orm import Session
from sqlalchemy import text
from psycopg2.extras import execute_values

import pandas as pd

from ..database import get_db
from ..modeling_services.anomaly_detection_model import AnomalyDetectionModel

from .schemas import *
from .exceptions import *
from .service import *
from .utils import *
from ..logger import logger


CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

router = APIRouter()


@router.post("/register")
def register(payload: RegisterPayload, db: Session = Depends(get_db)) -> Dict[str, Dict[str, int]]:
    """
    Registers nodes and sensors from the payload.
    Returns a mapping of node_hash to node_id and sensor_hash to sensor_id.
    Returns:
    {
        "nodes": {node_hash: node_id, ...},
        "sensors": {sensor_hash: sensor_id, ...}
    }
    """
    
    if not payload:
        logger.warning("Empty payload received in register endpoint")
        raise HTTPException(status_code=400, detail="Empty payload")

    return register_entities(payload, db)


@router.post("/dataIngest")
def data_ingest(payload: List[MeasurementPayload], db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    MeasurementPayload Pydantic model with:
    - sensor_hash: str
    - timestamp_utc: str
    - value: str
    """
    logger.info(f"Data ingest endpoint called with payload of length: {len(payload)}")

    if not payload:
        logger.warning("Empty payload received in data ingest endpoint")
        raise HTTPException(status_code=400, detail="Empty payload")

    return ingest_measurements(payload, db)

@router.post("/registerModel")
def register_model(request: CreateModelPayload, db: Session = Depends(get_db)):
    """
    Register a new model based on the request payload.
    Parameters:
    """
    logger.info(f"Register model endpoint called with request: {request}")

    if not request:
        logger.warning("Empty request received in register model endpoint")
        raise HTTPException(status_code=400, detail="Empty request")

    return create_model(request, db)

@router.post("/runModel")
async def run_model(payload: Dict, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Run specified model for a sensor with given configuration.
    Add new models to MODEL_REGISTRY.
    """
    logger.info(f"Run model endpoint called with payload: {payload}")
    
    if not payload:
        logger.warning("Empty payload received in run model endpoint")
        raise HTTPException(status_code=400, detail="Empty payload")

    return await model_results(payload, MODEL_REGISTRY, db)

@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    """
    Display all registered models with their details.
    """
    return get_models(db)

@router.get("/modelRuns")
def list_model_runs(db: Session = Depends(get_db)):
    """
    Display all registered model runs with their details.
    """
    return get_model_runs(db)

@router.get("/modelrun_logs/{run_id}")
def get_model_run_logs(run_id: int, db: Session = Depends(get_db)):
    """
    Display the logs for a specific model run.
    """
    return get_model_run_log(run_id, db)

@router.get("/models/{model_name}")
def list_model(model_name: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Display a specific registered model with its details.
    """
    return get_model(model_name, db)

@router.delete("/models")
def remove_models(db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Delete all models from the database. This will delete all models and associated configurations and results.
    """
    return delete_models(db)

@router.delete("/models/{model_name}")
def remove_model(model_name: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Delete model by name. This will delete the model and all associated configurations and results.
    """
    return delete_model(model_name, db)

@router.get("/nodes")
def list_nodes(db: Session = Depends(get_db)):
    """
    Display all registered nodes with their details.
    """
    return get_nodes(db)

@router.get("/nodes/{node_id}")
def list_node(node_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Display a specific registered node with its details.
    """
    return get_node(node_id, db)

@router.get("/sensors")
def list_sensors(db: Session = Depends(get_db)):
    """
    Display all registered sensors with their details.
    """
    return get_sensors(db)

@router.get("/sensors/active")
def list_active_sensors(db: Session = Depends(get_db)):
    """
    Display only sensors with status='active'.

    Prefer this over /sensors when the caller does not need inactive sensors
    (e.g. Active sensors KPI) — filtering happens in SQL, not in the client.
    """
    return get_sensors(db, status="active")

@router.get("/sensors/{sensor_id}")
def list_sensor(sensor_id: int, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Display a specific registered sensor with its details.
    """
    return get_sensor(sensor_id, db)

@router.get("/measurements")
def measurements(sensorIDs: Optional[List[int]] = Query(None), days: Optional[int] = 0, limit: Optional[int] = 10, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Retrieve the latest measurements for a given sensor.
    Parameters:
    - sensorIDs: List of sensor IDs to retrieve measurements for
    - days: Number of days to retrieve measurements for
    - limit: Number of latest measurements to retrieve (default: 10)
    """
    return get_measurements(sensorIDs, days, limit, db)