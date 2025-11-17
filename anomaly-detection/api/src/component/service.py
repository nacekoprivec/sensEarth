import os
import numpy as np
from api.src.component.schemas import DetectorCreateRequest
import main

import argparse
import tempfile
import json

import main

from .models import *
from ..database import get_db
from fastapi import Depends, HTTPException
from sqlalchemy.orm import Session
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
import pandas as pd
import time
from typing import Dict, Any, Optional
from enum import Enum
from .exceptions import *

CONFIG_DIR = os.path.abspath("configuration")
DATA_DIR = os.path.abspath("data")

def load_config(conf_name: str) -> Dict[str, Any]:
    """
    Load json config from /configuration folder
    """
    config_file = os.path.join(CONFIG_DIR, conf_name)
    try:
        with open(config_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        raise NotFoundException("Config file", conf_name)
    except json.JSONDecodeError:
        raise ConfigFileException(conf_name, "contains invalid JSON.")
    except Exception as e:
        raise InternalServerException(f"Unexpected error loading config: {e}")

def create_available_configs_enum() -> Enum:
    """
    Returns an Enum containing all JSON config filenames in CONFIG_DIR.
    """
    if not os.path.exists(CONFIG_DIR):
        raise InternalServerException(f"Config directory not found: {CONFIG_DIR}")

    files = [
        f for f in os.listdir(CONFIG_DIR)
        if os.path.isfile(os.path.join(CONFIG_DIR, f)) and f.endswith(".json")
    ]

    if not files:
        raise InternalServerException("No configuration files found in CONFIG_DIR.")

    enum_members = format_enum_members(files)
    return Enum("AvailableConfigs", enum_members)
    
def format_enum_members(files):
    """
    Converts filenames to valid Enum member dictionary.
    Example: "detector_config.json" -> {"DETECTOR_CONFIG": "detector_config.json"}
    """
    enum_dict = {}
    for f in files:
        name = os.path.splitext(f)[0].upper().replace(" ", "_")
        if not name.isidentifier():
            raise InternalServerException(f"Invalid config name '{name}' for Enum")
        enum_dict[name] = f
    return enum_dict

def create_json_config(body: dict, name: str) -> str:
    """Saves the configuration to a file named detector_{name}.json """
    os.makedirs(CONFIG_DIR, exist_ok=True)
    config_name = f"detector_{name}.json"
    config_path = os.path.join(CONFIG_DIR, config_name)
    try:
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(body, f, ensure_ascii=False, indent=2)
    except NotFoundException:
        raise 
    except Exception as e:
        print(f"Error creating config file {config_path}: {e}")
        raise

    return config_name

# CREATE anomaly detector
def create_anomaly_detector(request: Dict, db: Session):
    """Creates and sets anomaly detector status to inactive"""
    print("Creating detector with request:", request)
    try:
        if has_custom_config(request): 
            config_data = {
                "anomaly_detection_alg": request["config_data"]["anomaly_detection_alg"],
                "anomaly_detection_conf": request["config_data"]["anomaly_detection_conf"]
            }
            print("Creating detector with custom config", config_data)
        elif request["config_name"]:
            config_data = load_config(request["config_name"])
        else:
            raise ValueError("config_name or anomaly_detection_alg + anomaly_detection_conf must be provided")

        detector_conf_name = create_json_config(config_data, request["name"])

        detector = AnomalyDetector(
            name=request["name"],
            description=request["description"],
            updated_at=datetime.datetime.now(datetime.timezone.utc),
            status="inactive",
            config_name=detector_conf_name,
            config=json.dumps(config_data)
        )
        db.add(detector)
        db.commit()        
        db.refresh(detector)
  
        return detector

    except NotFoundException:
        db.rollback()
        raise 
    except JSONDecodeException:
        db.rollback()
        raise 
    except Exception as e:
        db.rollback()
        raise

def has_custom_config(request: Dict) -> bool:
    """
    Returns True if both anomaly_detection_alg and anomaly_detection_conf are provided.
    """
    cfg = request.get("config_data")
    if isinstance(cfg, dict):
        return bool(cfg.get("anomaly_detection_alg")) and bool(cfg.get("anomaly_detection_conf"))
    return False

# READ anomaly detectors

def get_anomaly_detector(db: Session, detector_id: int):
    try:
        return db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
    except Exception as e:
        raise InternalServerException(f"Failed to fetch anomaly detector {detector_id}: {e}")

def get_anomaly_detectors(db: Session, skip: int = 0, limit: int = 50):
    try:
        return db.query(AnomalyDetector).offset(skip).limit(limit).all()
    except Exception as e:
        raise InternalServerException(f"Failed to fetch anomaly detectors: {e}")

# UPDATE anomaly detectors

def update_anomaly_detector(db: Session, detector_id: int, name: Optional[str] = None, description: Optional[str] = None,) -> Optional[AnomalyDetector]:
    try: 
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if not detector:
            return None
        if name: # "" not allowed 
            detector.name = name
        if description is not None:
            detector.description = description
        detector.updated_at = datetime.now(timezone.utc)

        db.commit()
        db.refresh(detector)
        return detector
    except Exception as e:
        db.rollback() 
        raise InternalServerException(f"Failed to update anomaly detector: {e}")


# Delete anomaly detectors

def delete_anomaly_detector(detector_id: int, db: Session):
    """Deletes specific detector and removes its detector_{name}.json file"""
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if not detector:
            raise DetectorNotFoundException

        if detector.config_name.startswith("detector_"):
            config_path = os.path.join(CONFIG_DIR, detector.config_name)
            if os.path.exists(config_path):
                os.remove(config_path)
        db.delete(detector)
        db.commit()
        return detector
    except Exception as e:
        db.rollback()
        raise InternalServerException(f"Failed to update anomaly detector: {e}")

def delete_all_detectors(db: Session):
    """Deletes all detectors and removes their config files"""
    try:
        detectors = db.query(AnomalyDetector).all()
        for detector in detectors:
            if detector.config_name and detector.config_name.startswith("detector_"):
                config_path = os.path.join(CONFIG_DIR, detector.config_name)
                if os.path.exists(config_path):
                    os.remove(config_path)
            db.delete(detector)
        db.commit()
        return len(detectors)
    except Exception as e:
        db.rollback()
        raise InternalServerException(f"Failed to delete all anomaly detectors: {e}")


# set anomaly detector active/inactive
def set_detector_status(detector_id: int, status: str, db: Session):
    try:
        detector = db.query(AnomalyDetector).filter(AnomalyDetector.id == detector_id).first()
        if detector:
            detector.status = status
            db.add(detector)
            db.commit()
            db.refresh(detector)
        return detector
    except Exception as e:
        db.rollback()
        raise InternalServerException(f"Failed to set detector status: {e}")


# Format HH:MM:SS
def format_seconds(seconds: float) -> str:
    if seconds is None:
        return "N/A"
    seconds = round(seconds)
    return seconds

#deprecated
# def delete_all_logs(db: Session):
#     try:
#         num_deleted = db.query(Log).delete()
#         db.commit()
#         return num_deleted
#     except Exception as e:
#         db.rollback()
#         print(f"Error deleting all logs: {e}")
#         return 0

# def delete_log(log_id: int, db: Session):
#     try:
#         log = db.query(Log).filter(Log.id == log_id).first()
#         if log:
#             db.delete(log)
#             db.commit()
#         return log
#     except Exception as e:
#         db.rollback()
#         print(f"Error deleting log: {e}")
#         return None
    
# def get_logs(db: Session, skip: int = 0, limit: int = 10):
#     return db.query(Log).offset(skip).limit(limit).all()

# def get_log(db: Session, log_id: int):
#     return db.query(Log).filter(Log.id == log_id).first()