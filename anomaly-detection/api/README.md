# API for Anomaly Detection Algorithms

This API provides endpoints for managing sensor data, registering and running anomaly detection and forecasting models, and retrieving results in the SensEarth system.

## Overview

The API handles:
- Registration of nodes and sensors.
- Ingestion of measurement data.
- Model management (registration, execution, listing).
- Retrieval of nodes, sensors, measurements, and model results.

It integrates with the monitoring system for logging events, metrics, and heartbeats.

## Running the API

The API runs as part of the `middleware-api` service in Docker Compose. Ensure the environment variables are set (e.g., `CORE_DB_URL`, `MONITORING_API_URL`).

To start: `docker compose up middleware-api`

## Endpoints

### Data Management

#### POST /register
Registers nodes and sensors.

**Request Body:**
```json
{
  "nodes": [
    {
      "node_label": "Station1",
      "node_serial": "12345",
      "longitude": 14.5,
      "latitude": 46.0,
      "altitude": 300.0
    }
  ],
  "sensors": [
    {
      "node_hash": "...",
      "sensor_label": "TempSensor",
      "sensor_name": "Temperature Sensor",
      "longitude": 14.5,
      "latitude": 46.0,
      "altitude": 300.0,
      "sensor_type": {
        "name": "air_temperature",
        "phenomenon": "Air Temperature",
        "unit": "°C",
        "value_min": -50,
        "value_max": 60
      }
    }
  ]
}
```

**Response:**
```json
{
  "nodes": {"hash1": 1},
  "sensors": {"hash2": 2}
}
```

#### POST /dataIngest
Ingests measurement data.

**Request Body:**
```json
[
  {
    "sensor_hash": "hash",
    "timestamp_utc": "2023-01-01T12:00:00Z",
    "value": "25.5"
  }
]
```

**Response:** Success confirmation.

## Models

- **AnomalyDetectionModel**: Detects anomalies using algorithms like BorderCheck, EMA.
- **ForecastModel**: Provides forecasting capabilities.

Models inherit from `BaseModel` and emit monitoring events during execution.

## Usage Examples

### Model Management

#### POST /registerModel
Registers a new model.

**Request Body:**
```json
{
  "model_name": "border_check_AD",
  "model_description": "Detects anomalies in sensor data",
  "model_type": "anomaly_detection",
  "model_parameters": {
    "anomaly_detection_alg": ["BorderCheck()"],
    "anomaly_detection_conf": [
      {
        "input_vector_size": 1,
        "warning_stages": [2.5, 0.0],
        "UL": 3.0,
        "LL": -0.4,
        "output": ["TerminalOutput()"],
        "output_conf": [{}]
      }
    ]
  },
  "sensor_id_list": [1, 2, 3]
}
```

#### POST /runModel
Runs a model on sensor data.

**Request Body:**
```json
{
  "model_name": "border_check_AD",
  "sliding_window_size": 100,
  "sensor_id_list": [10, 20, 30],
  "parameters": {
    "anomaly_detection_alg": ["EMA()"],
    "anomaly_detection_conf": [
      {
        "input_vector_size": 1,
        "N": 3,
        "LL": -0.45,
        "UL": 0.55,
        "warning_stages": [0.3, -0.2],
        "output": ["TerminalOutput()"],
        "output_conf": [{}]
      }
    ]
  }
}
```

For more details, refer to the configuration files in `configuration/` and data in `data/`.

Project structure based on https://github.com/zhanymkanov/fastapi-best-practices?tab=readme-ov-file#project-structure

All endpoints can be found in MIDDLEWARE_API docs (eg. http://localhost:5006/docs#/) 