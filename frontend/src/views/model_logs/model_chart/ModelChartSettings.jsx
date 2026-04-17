import React, { useEffect, useMemo, useState } from "react";
import api from "../../../api";
import SensorChartModels from "./SensorChart_models";
import { Row, Col, Card, Table, Spinner } from "react-bootstrap";


function formatRunLabel(run) {
  const started = run.started_at ? new Date(run.started_at).toLocaleString() : "unknown start";
  const status = run.status ?? "unknown";
  return `#${run.run_id} · model ${run.model_id} · ${status} · ${started}`;
}

export default function ModelChartSettings({ allSensors }) {
  const [runs, setRuns] = useState([]);
  const [runsLoading, setRunsLoading] = useState(false);
  const [runsError, setRunsError] = useState(null);

  const [selectedRunId, setSelectedRunId] = useState("");
  const [logsLoading, setLogsLoading] = useState(false);
  const [logsError, setLogsError] = useState(null);
  const [measurements, setMeasurements] = useState([]);

  const sensorsById = useMemo(() => {
    const map = new Map();
    (allSensors || []).forEach(s => map.set(Number(s.sensor_id), s));
    return map;
  }, [allSensors]);

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
        if (data.length && selectedRunId === "") setSelectedRunId(String(data[0].run_id));
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
  }, [selectedRunId]);

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
        const normalized = rows.map(r => {
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
    <>
      <Card className="flat-card">
        <Card.Body>
          <div className="border-bottom d-flex justify-content-between align-items-center mb-3">
            <h3>Model run overview</h3>
            <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
              <label className="settings-label" style={{ marginBottom: 0 }}>
                Run
              </label>
              <select
                className="form-select"
                style={{ width: 420 }}
                value={selectedRunId}
                onChange={e => setSelectedRunId(e.target.value)}
                disabled={runsLoading || !runs.length}
              >
                {runs.map(r => (
                  <option key={r.run_id} value={String(r.run_id)}>
                    {formatRunLabel(r)}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {runsError && <div className="text-danger mb-2">{runsError}</div>}
          {logsError && <div className="text-danger mb-2">{logsError}</div>}
          {(runsLoading || logsLoading) && <div className="text-muted mb-2">Loading…</div>}

          <SensorChartModels measurements={measurements} />
        </Card.Body>
      </Card>
    </>
  );
}