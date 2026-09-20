-- ===============================
-- 0. Drop old tables 
-- ===============================
DROP TABLE IF EXISTS sensor_measurement CASCADE;
DROP TABLE IF EXISTS sensor CASCADE;
DROP TABLE IF EXISTS sensor_node CASCADE;
DROP TABLE IF EXISTS sensor_type CASCADE;
DROP TABLE IF EXISTS model_inference CASCADE;
DROP TABLE IF EXISTS model_run CASCADE;

-- ===============================
-- 1. Enable extensions
-- ===============================
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ===============================
-- 2. SENSOR TYPE
-- ===============================
CREATE TABLE IF NOT EXISTS sensor_type (
    sensor_type_id SERIAL PRIMARY KEY,
    name VARCHAR(64) UNIQUE NOT NULL,
    phenomenon VARCHAR(128) NOT NULL,
    unit VARCHAR(32) NOT NULL,
    value_min DOUBLE PRECISION,
    value_max DOUBLE PRECISION,
    description TEXT
);

-- ===============================
-- 3. SENSOR NODE
-- ===============================
CREATE TABLE IF NOT EXISTS sensor_node (
    node_id SERIAL PRIMARY KEY,
    node_label VARCHAR(64) NOT NULL,
    node_hash VARCHAR(128) UNIQUE NOT NULL,
    location GEOGRAPHY(PointZ, 4326),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(32) DEFAULT 'active',
    description TEXT
);

-- ===============================
-- 4. SENSOR
-- ===============================
CREATE TABLE IF NOT EXISTS sensor (
    sensor_id SERIAL PRIMARY KEY,
    sensor_label VARCHAR(64) NOT NULL,
    sensor_hash VARCHAR(128) UNIQUE NOT NULL,
    node_id INT REFERENCES sensor_node(node_id) ON DELETE SET NULL,
    sensor_type_id INT REFERENCES sensor_type(sensor_type_id),
    description TEXT,
    location GEOGRAPHY(PointZ, 4326),
    last_seen TIMESTAMPTZ DEFAULT NOW(),
    status VARCHAR(32) DEFAULT 'active',
    metadata JSONB, 
    UNIQUE(node_id, sensor_hash) 
);

-- ===============================
-- 5. SENSOR MEASUREMENT
-- ===============================
CREATE TABLE IF NOT EXISTS sensor_measurement (
    sensor_id INT REFERENCES sensor(sensor_id) ON DELETE CASCADE,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    source VARCHAR(64) NOT NULL DEFAULT 'live',
    PRIMARY KEY(sensor_id, timestamp_utc)
);

CREATE INDEX IF NOT EXISTS idx_sensor_measurement_source
    ON sensor_measurement (source);

-- ===============================
-- 6. Convert sensor_measurement to Timescale hypertable
-- ===============================
SELECT create_hypertable(
    'sensor_measurement',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '7 day'
);

-- ===============================
-- 7. Models
-- ===============================

CREATE TABLE IF NOT EXISTS model (
    model_id SERIAL PRIMARY KEY,
    name VARCHAR(64) UNIQUE NOT NULL,
    description TEXT,
    model_type VARCHAR(32) NOT NULL,
    parameters JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE model_sensor (
    model_id INT NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
    sensor_id INT NOT NULL REFERENCES sensor(sensor_id) ON DELETE CASCADE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (model_id, sensor_id)
);

CREATE TABLE IF NOT EXISTS model_run (
    run_id SERIAL PRIMARY KEY,
    model_id INT NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
    started_at TIMESTAMPTZ DEFAULT NOW(),
    finished_at TIMESTAMPTZ,
    status VARCHAR(32),
    notes TEXT
);

CREATE TABLE IF NOT EXISTS model_inference (
    inference_id SERIAL PRIMARY KEY,
    run_id INT NOT NULL REFERENCES model_run(run_id) ON DELETE CASCADE,
    model_id INT NOT NULL REFERENCES model(model_id) ON DELETE CASCADE,
    sensor_id INT NOT NULL REFERENCES sensor(sensor_id) ON DELETE CASCADE,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION,
    inference_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ===============================
-- Foreign key relationships:
-- sensor_type 1—n sensor
-- sensor_node 1—n sensor
-- sensor 1—n sensor_measurement
-- sensor 1—n sensor_measurement
-- model 1—n model_sensor
-- sensor 1—n model_sensor