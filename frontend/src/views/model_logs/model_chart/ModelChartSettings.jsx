import React, { useEffect, useMemo, useState } from "react";
import { Card, Form, Spinner } from "react-bootstrap";
import api from "../../../api";
import SensorChartModels from "./SensorChart_models";

function formatRunLabel(run) {
  const started = run.started_at ? new Date(run.started_at).toLocaleString() : "unknown start";
  const status = run.status ?? "unknown";
  return `#${run.run_id} - model ${run.model_id} - ${status} - ${started}`;
}

const statusBadgeClass = (status) => {
  switch (status) {
    case "completed":
      return "model-run-status-badge model-run-status-badge--completed";
    case "running":
      return "model-run-status-badge model-run-status-badge--running";
    case "failed":
    case "error":
      return "model-run-status-badge model-run-status-badge--error";
    default:
      return "model-run-status-badge";
  }
};

export default function ModelChartSettings({ allSensors, refreshKey }) {
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState(null);
  const [selectedRunId, setSelectedRunId] = useState("");
  const [logsLoading, setLogsLoading] = useState(false);
  const [logsError, setLogsError] = useState(null);
  const [measurements, setMeasurements] = useState([]);

  const sensorsById = useMemo(() => {
    const map = new Map();
    (allSensors || []).forEach((s) => map.set(Number(s.sensor_id), s));
    return map;
  }, [allSensors]);

  const selectedRun = useMemo(
    () => runs.find((r) => String(r.run_id) === selectedRunId) ?? null,
    [runs, selectedRunId]
  );

  useEffect(() => {
    let cancelled = false;
    async function loadRuns() {
      setRunsLoading(true);
      setRunsError(null);
      try {
        const res = await api.get("/modelRuns");
        if (cancelled) return;
        const data = Array.isArray(res.data) ? res.data : [];
        setRuns(data);
        if (data.length && !selectedRunId) setSelectedRunId(String(data[0].run_id));
      } catch (e) {
        if (cancelled) return;
        setRuns([]);
        setRunsError("Failed to load model runs.");
      } finally {
        if (!cancelled) setRunsLoading(false);
      }
    }
    loadRuns();
    return () => {
      cancelled = true;
    };
  }, [refreshKey]);

  useEffect(() => {
    if (!selectedRunId) return;
    let cancelled = false;
    async function loadLogs() {
      setLogsLoading(true);
      setLogsError(null);
      try {
        const res = await api.get(`/modelrun_logs/${encodeURIComponent(selectedRunId)}`);
        if (cancelled) return;
        const rows = Array.isArray(res.data) ? res.data : [];
        const normalized = rows.map((r) => {
          const sensorId = Number(r.sensor_id);
          const sensor = sensorsById.get(sensorId);
          const inferenceMessage = r.inference_message ?? "";
          return {
            run_id: r.run_id,
            model_id: r.model_id,
            sensor_id: sensorId,
            sensor_label: sensor?.sensor_label ?? `sensor_${sensorId}`,
            timestamp_utc: r.timestamp_utc,
            value: r.value,
            inference_message: inferenceMessage,
            is_anomaly: inferenceMessage !== "OK",
          };
        });
        setMeasurements(normalized);
      } catch (e) {
        if (cancelled) return;
        setMeasurements([]);
        setLogsError("Failed to load model run logs.");
      } finally {
        if (!cancelled) setLogsLoading(false);
      }
    }
    loadLogs();
    return () => {
      cancelled = true;
    };
  }, [selectedRunId, sensorsById]);

  return (
      <Card className="flat-card dashboard-component model-run-overview">
      <Card.Body className="d-flex flex-column">
        <div className="model-run-overview__header">
          <h5 className="model-panel__title mb-0">Model run overview</h5>
          <div className="model-run-overview__controls">
            {selectedRun?.status && (
              <span className={statusBadgeClass(selectedRun.status)}>{selectedRun.status}</span>
            )}
            <span className="model-run-overview__run-label">RUN</span>
            <Form.Select
              size="sm"
              className="model-run-overview__select"
              value={selectedRunId}
              onChange={(e) => setSelectedRunId(e.target.value)}
              disabled={runsLoading || !runs.length}
            >
              {runs.map((r) => (
                <option key={r.run_id} value={String(r.run_id)}>
                  {formatRunLabel(r)}
                </option>
              ))}
            </Form.Select>
          </div>
        </div>

        {runsError && <div className="text-danger small mb-2">{runsError}</div>}
        {logsError && <div className="text-danger small mb-2">{logsError}</div>}
        {(runsLoading || logsLoading) && (
          <div className="text-muted small mb-2">
            <Spinner animation="border" size="sm" className="me-2" />
            Loading…
          </div>
        )}

        <div className="model-run-overview__chart">
          <SensorChartModels measurements={measurements} />
        </div>
      </Card.Body>
    </Card>
  );
}
