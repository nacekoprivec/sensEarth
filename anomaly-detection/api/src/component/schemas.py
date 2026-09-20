from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

class NodePayload(BaseModel):
    node_label: str
    node_hash: str
    longitude: float
    latitude: float
    altitude: Optional[float] = None
    description: Optional[str] = ""

class SensorTypePayload(BaseModel):
    name: str
    phenomenon: str
    unit: str
    value_min: Optional[float]
    value_max: Optional[float]

class SensorPayload(BaseModel):
    sensor_label: str
    sensor_hash: str
    node_hash: str
    sensor_type: SensorTypePayload
    longitude: Optional[float]
    latitude: Optional[float]
    altitude: Optional[float]
    sensor_description: Optional[str] = ""
    # Merge-only on upsert; send null for a key to remove it, e.g. {"sifra": null}
    metadata: Optional[Dict[str, Any]] = None

class RegisterPayload(BaseModel):
    nodes: List[NodePayload]
    sensors: List[SensorPayload]

class MeasurementPayload(BaseModel):
    sensor_hash: str
    timestamp_utc: str  
    value: Optional[str]
    source: Optional[str] = "live"

class DeleteBySourcePayload(BaseModel):
    source: str

class dataIngestPayload(BaseModel):
    measurements: List[MeasurementPayload]

class runModelPayload(BaseModel):
    sensor_id: str
    algorithm_name: str
    model_type: str
    sliding_window_size: Optional[int] = 100

class CreateModelPayload(BaseModel):
    model_name: str 
    model_description: Optional[str]
    model_parameters: Optional[Dict]
    sensor_id_list: Optional[List[int]]
    model_type: Optional[str]

class RunModelPayload(BaseModel):
    model_name: str
    sensor_id_list: List[int]
    parameters: Any
    model_type: str
    sliding_window_size: Optional[int] = 100
