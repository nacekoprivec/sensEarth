-- ===============================
-- 0. Drop old tables 
-- ===============================
DROP TABLE IF EXISTS sensor_measurement CASCADE;
DROP TABLE IF EXISTS sensor CASCADE;
DROP TABLE IF EXISTS sensor_node CASCADE;
DROP TABLE IF EXISTS sensor_type CASCADE;

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
    name VARCHAR(128) NOT NULL,
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
    name VARCHAR(128) NOT NULL,
    description TEXT,
    metadata JSONB
);

-- ===============================
-- 4. SENSOR
-- ===============================
CREATE TABLE IF NOT EXISTS sensor (
    sensor_id SERIAL PRIMARY KEY,
    node_id INT REFERENCES sensor_node(node_id) ON DELETE SET NULL,
    sensor_type_id INT REFERENCES sensor_type(sensor_type_id),
    name VARCHAR(128) NOT NULL,
    description TEXT,
    location GEOGRAPHY(Point, 4326), -- requires PostGIS
    metadata JSONB,
    UNIQUE(node_id, name)
);

-- ===============================
-- 5. SENSOR MEASUREMENT
-- ===============================
CREATE TABLE IF NOT EXISTS sensor_measurement (
    sensor_id INT REFERENCES sensor(sensor_id) ON DELETE CASCADE,
    timestamp_utc TIMESTAMPTZ NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    metadata JSONB,
    PRIMARY KEY(sensor_id, timestamp_utc)
);

-- ===============================
-- 6. Convert sensor_measurement to Timescale hypertable
-- ===============================
SELECT create_hypertable(
    'sensor_measurement',
    'timestamp_utc',
    chunk_time_interval => INTERVAL '1 day'
);

-- ===============================
-- Foreign key relationships:
-- sensor_type 1—n sensor
-- sensor_node 1—n sensor
-- sensor 1—n sensor_measurement
