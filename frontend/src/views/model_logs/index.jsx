import React, { useEffect, useMemo, useState } from "react";
import { Row, Col, Card, Table, Spinner } from "react-bootstrap";
import api from "../../api";
import ModelChartSettings from "./model_chart/ModelChartSettings";
import ModelsDashboard from "./models";
import ModelLogs from "./model_logs";
// -----------------------|| MODEL LOGS ||-----------------------//

export default function Models() {
  const [models, setModels] = useState([]);
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [expandedModelIds, setExpandedModelIds] = useState(new Set());
  const [selectedRun, setSelectedRun] = useState(null);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [runLogs, setRunLogs] = useState([]);
  const [logsError, setLogsError] = useState(null);
  const [selectedModel, setSelectedModel] = useState(null);
  const [allSensors, setAllSensors] = useState([]); // [{id, label}]
  const [modelsUpdated, setModelsUpdated] = useState(0);


  const fetchData = async () => {
    setLoading(true);
    try {
      const [modelsRes, runsRes] = await Promise.all([
        api.get("/models"),
        api.get("/modelRuns"),
      ]);

      const modelsData = Array.isArray(modelsRes.data) ? modelsRes.data : [];
      const runsData = Array.isArray(runsRes.data) ? runsRes.data : [];

      setModels(modelsData);
      setRuns(runsData);
    } catch (e) {
      console.error("Failed to fetch models or runs:", e);
      setModels([]);
      setRuns([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchData();
  }, []);

  useEffect(() => {
    const runId = selectedRun?.run_id;
    if (!runId) {
      setRunLogs([]);
      setLogsError(null);
      setLoadingLogs(false);
      return;
    }

    let cancelled = false;

    const fetchSensorsAll = async () => {
      try {
        const res = await api.get("/sensors");
        setAllSensors(res.data);
        console.log("Fetched all sensors:", res.data);
      } catch (error) {
        console.error("Failed to fetch sensors:", error);
        setAllSensors([]);
      }
      setLoading(false);
    };


    const fetchLogs = async () => {
      setLoadingLogs(true);
      setLogsError(null);
      try {
        const res = await api.get(`/modelrun_logs/${encodeURIComponent(runId)}`);
        const data = Array.isArray(res.data) ? res.data : [];
        if (!cancelled) setRunLogs(data);
      } catch (e) {
        console.error("Failed to fetch run logs:", e);
        if (!cancelled) {
          setRunLogs([]);
          setLogsError("Failed to load logs for this run.");
        }
      }
      if (!cancelled) setLoadingLogs(false);
    };

    fetchLogs();

    return () => {
      cancelled = true;
    };
  }, [selectedRun?.run_id]);

  const runsByModelId = useMemo(() => {
    const map = new Map();
    for (const run of runs) {
      const mid = run.model_id;
      if (mid == null) continue;
      if (!map.has(mid)) map.set(mid, []);
      map.get(mid).push(run);
    }
    // sort runs newest first
    for (const list of map.values()) {
      list.sort((a, b) => {
        const aStart = a.started_at ? new Date(a.started_at).getTime() : 0;
        const bStart = b.started_at ? new Date(b.started_at).getTime() : 0;
        return bStart - aStart;
      });
    }
    return map;
  }, [runs]);

  const toggleModelExpanded = (modelId) => {
    setExpandedModelIds((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) {
        next.delete(modelId);
      } else {
        next.add(modelId);
      }
      return next;
    });
  };

  const formatDateTime = (value) => {
    if (!value) return "—";
    try {
      return new Date(value).toLocaleString();
    } catch {
      return String(value);
    }
  };

  const formatRunLabel = (run) => {
    if (run.run_id != null) return `Run #${run.run_id}`;
    if (run.started_at) return `Run ${formatDateTime(run.started_at)}`;
    return "Run";
  };

  const sortedRunLogs = useMemo(() => {
    const copy = Array.isArray(runLogs) ? [...runLogs] : [];
    copy.sort((a, b) => {
      const at = a.timestamp_utc ? new Date(a.timestamp_utc).getTime() : 0;
      const bt = b.timestamp_utc ? new Date(b.timestamp_utc).getTime() : 0;
      return bt - at;
    });
    return copy;
  }, [runLogs]);

  return (
    <>
      <div className="dashboard-grid">
        <ModelLogs refreshKey={modelsUpdated} />
        <ModelChartSettings allSensors={allSensors} />
        <ModelsDashboard setModelsUpdated={setModelsUpdated} />
      </div>
    </>
  );
}
