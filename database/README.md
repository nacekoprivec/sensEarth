# Databases
The SensEarth database is a **PostgreSQL + PostGIS + TimescaleDB** system for storing sensor networks, time-series data, and ML model outputs.

## Purpose
- Sensor infrastructure (nodes, sensors, types)
- Time-series measurements
- Machine learning models
- Model outputs (runs + inferences)

## Main Entities

### Sensor Types
Defines what is measured (e.g. temperature, diesel price).
Includes unit, name, and value range.

---

### Sensor Nodes
Physical locations (stations).
Contains:
- unique hash
- geolocation (PostGIS)
- status

---

### Sensors
Data sources attached to nodes.
Each sensor:
- belongs to a node
- has a sensor type
- produces measurements over time

---

### Sensor Measurements
Time-series data (core dataset).
Stores:
- sensor_id
- timestamp (UTC)
- value

Stored in a **TimescaleDB hypertable** for performance.

---

### Models
Machine learning model definitions.
Includes:
- type
- parameters (JSON)
- metadata

---

### Model Runs
Execution logs of models.
Tracks:
- start/end time
- status

---

### Model Inference
Results produced by models.
Includes:
- sensor reference
- timestamp
- predicted value

---

## Key Relationships
sensor_type -> sensor (1:N)
node -> sensor (1:N)
sensor -> measurements (1:N)
model -> runs (1:N)
run -> inference (1:N)

Refer to `./database_create.sql` for details

# Export data for portability 
Sensearth db
```bash 
docker exec sensearth-core-db pg_dump -U postgres -Fc sensearth_db > init_db/sensearth.dump
```

Monitoring db
```bash 
docker exec sensearth-monitoring-db pg_dump -U postgres -Fc monitoring_db > init_db/monitoring.dump
```

