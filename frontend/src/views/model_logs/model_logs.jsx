import React, { useEffect, useMemo, useState } from "react";
import { Card, Table, Spinner } from "react-bootstrap";
import api from "../../api";

// -----------------------|| MODEL LOGS ||-----------------------//

export default function ModelLogs( { refreshKey } ) {
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
  }, [refreshKey]);

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
    <div className="model-logs-layout d-flex gap-3 h-100">
        <Card className="flat-card dashboard-component">
          <Card.Body className="overflow-auto">
            <div className="border-bottom d-flex justify-content-between align-items-center mb-3">
              <h3 className="mb-0">Model logs</h3>
            </div>

            {loading ? (
              <div className="text-center my-4">
                <Spinner animation="border" />
              </div>
            ) : models.length === 0 ? (
              <div className="text-muted text-center my-4">
                No models found.
              </div>
            ) : (
              <Table hover responsive size="sm" className="align-middle mb-0">
                <thead className="table-light">
                  <tr>
                    <th style={{ width: "40%" }}>Model</th>
                    <th style={{ width: "25%" }}>Type</th>
                    <th style={{ width: "20%" }}>Runs</th>
                    <th style={{ width: "15%" }} className="text-end">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {models.map((model) => {
                    const modelRuns =
                      runsByModelId.get(model.model_id) || [];
                    const isExpanded = expandedModelIds.has(model.model_id);

                    return (
                      <React.Fragment key={model.model_id}>
                        <tr
                          className="cursor-pointer"
                          onClick={() => {
                            toggleModelExpanded(model.model_id);
                            setSelectedModel(model);
                          }}
                        >
                          <td>
                            <div className="fw-semibold text-truncate">
                              {model.name}
                            </div>
                            {model.description && (
                              <div className="text-muted small text-truncate">
                                {model.description}
                              </div>
                            )}
                          </td>
                          <td>
                            <span className="badge bg-light text-dark border">
                              {model.model_type || "N/A"}
                            </span>
                          </td>
                          <td>
                            {modelRuns.length === 0 ? (
                              <span className="text-muted small">
                                No runs
                              </span>
                            ) : (
                              <span className="small">
                                {modelRuns.length} run
                                {modelRuns.length > 1 ? "s" : ""}
                              </span>
                            )}
                          </td>
                          <td className="text-end">
                            <span className="text-primary small">
                              {isExpanded ? "Hide runs ▲" : "Show runs ▼"}
                            </span>
                          </td>
                        </tr>

                        {isExpanded && (
                          <tr>
                            <td colSpan={4} className="bg-light">
                              {modelRuns.length === 0 ? (
                                <div className="text-muted small">
                                  This model has no recorded runs.
                                </div>
                              ) : (
                                <Table
                                  hover
                                  size="sm"
                                  className="mb-0 bg-white"
                                >
                                  <thead>
                                    <tr>
                                      <th style={{ width: "40%" }}>
                                        Run
                                      </th>
                                      <th style={{ width: "30%" }}>
                                        Start
                                      </th>
                                      <th style={{ width: "30%" }}>
                                        Finish
                                      </th>
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {modelRuns.map((run) => (
                                      <tr
                                        key={run.run_id}
                                        className="cursor-pointer"
                                        onClick={() => setSelectedRun(run)}
                                      >
                                        <td>{formatRunLabel(run)}</td>
                                        <td>{formatDateTime(run.started_at)}</td>
                                        <td>
                                          {formatDateTime(run.finished_at)}
                                        </td>
                                      </tr>
                                    ))}
                                  </tbody>
                                </Table>
                              )}
                            </td>
                          </tr>
                        )}
                      </React.Fragment>
                    );
                  })}
                </tbody>
              </Table>
            )}

            <hr className="my-4" />
            <div className="mt-2">
              <h5 className="mb-2 border-bottom pb-2">Selected run</h5>
              {!selectedRun ? (
                <div className="text-muted small">
                  Click on a run to see its details.
                </div>
              ) : (
                <div className="small">
                  <div className="mb-1">
                    <span className="text-muted">Run:</span>{" "}
                    <span className="fw-semibold">
                      {formatRunLabel(selectedRun)}
                    </span>
                  </div>
                  <div className="mb-1">
                    <span className="text-muted">Status:</span>{" "}
                    <span className="fw-semibold">
                      {selectedRun.status || "N/A"}
                    </span>
                  </div>
                  <div className="mb-1">
                    <span className="text-muted">Start:</span>{" "}
                    <span className="fw-semibold">
                      {formatDateTime(selectedRun.started_at)}
                    </span>
                  </div>
                  <div className="mb-1">
                    <span className="text-muted">Finish:</span>{" "}
                    <span className="fw-semibold">
                      {formatDateTime(selectedRun.finished_at)}
                    </span>
                  </div>

                  <div className="mt-3 scroll-area">
                    <div className="fw-semibold mb-2">Run logs</div>
                    {loadingLogs ? (
                      <div className="text-muted small mt-1">
                        <Spinner animation="border" size="sm" className="me-2" />
                        Loading…
                      </div>
                    ) : logsError ? (
                      <div className="text-danger small">{logsError}</div>
                    ) : sortedRunLogs.length === 0 ? (
                      <div className="text-muted small">No logs found for this run.</div>
                    ) : (
                      <Table hover responsive size="sm" className="align-middle mb-0">
                        <thead className="table-light">
                          <tr>
                            <th style={{ width: "25%" }}>Timestamp</th>
                            <th style={{ width: "15%" }}>Sensor</th>
                            <th style={{ width: "15%" }}>Value</th>
                            <th style={{ width: "45%" }}>Message</th>
                          </tr>
                        </thead>
                        <tbody>
                          {sortedRunLogs.slice(0, 200).map((row) => (
                            <tr key={row.inference_id ?? `${row.sensor_id}-${row.timestamp_utc}`}>
                              <td>{formatDateTime(row.timestamp_utc)}</td>
                              <td>{row.sensor_id ?? "—"}</td>
                              <td>{row.value ?? "—"}</td>
                              <td className="text-truncate" style={{ maxWidth: 0 }}>
                                {row.inference_message ?? "—"}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </Table>
                    )}
                    {sortedRunLogs.length > 200 && (
                      <div className="text-muted small mt-1">
                        Showing first 200 log rows.
                      </div>
                    )}
                  </div>
                </div>
              )}
            </div>
          </Card.Body>
        </Card>
        <Card className="flat-card dashboard-component">
            <Card.Body className="overflow-auto d-flex flex-column">
              <h5 className="mb-3 border-bottom pb-2">Model parameters</h5>

              {!selectedModel ? (
                <div className="text-muted small">
                  Click a model to view its parameters.
                </div>
              ) : (
                <>
                  <div className="mb-2 fw-semibold">
                    {selectedModel.name}
                  </div>
                  <div className="small text-muted mb-2">
                    Type: {selectedModel.model_type || "N/A"}
                  </div>
                  <hr className="my-2" />
                  {selectedModel.parameters ? (
                    <pre className="model-logs-params-pre mb-0 flex-grow-1">
                      {JSON.stringify(selectedModel.parameters, null, 2)}
                    </pre>
                  ) : (
                    <div className="text-muted small">
                      No parameters found.
                    </div>
                  )}
                </>
              )}
            </Card.Body>
        </Card>
    </div>
  );
}
