-- Add provenance tag for live vs historic measurement rollback.
-- Safe to run on an existing Timescale hypertable.

ALTER TABLE sensor_measurement
    ADD COLUMN IF NOT EXISTS source VARCHAR(64) NOT NULL DEFAULT 'live';

CREATE INDEX IF NOT EXISTS idx_sensor_measurement_source
    ON sensor_measurement (source);
