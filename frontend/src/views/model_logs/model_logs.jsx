import React, { useEffect, useMemo, useState } from "react";
import { Card, Table, Spinner } from "react-bootstrap";
import FeatherIcon from "feather-icons-react";
import api from "../../api";

export default function ModelLogs({ refreshKey, selectedModel }) {
  const [runs, setRuns] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedRun, setSelectedRun] = useState(null);
  const [loadingLogs, setLoadingLogs] = useState(false);
  const [runLogs, setRunLogs] = useState([]);
  const [logsError, setLogsError] = useState(null);

  useEffect(() => {
    setLoading(true);
    api
      .get("/modelRuns")
      .then((res) => setRuns(Array.isArray(res.data) ? res.data : []))
      .catch((e) => {
        console.error("Failed to fetch runs:", e);
        setRuns([]);
      })
      .finally(() => setLoading(false));
  }, [refreshKey]);

  const modelRuns = useMemo(() => {
    if (!selectedModel) return [];
    return runs
      .filter((r) => r.model_id === selectedModel.model_id)
      .sort((a, b) => new Date(b.started_at || 0) - new Date(a.started_at || 0));
  }, [runs, selectedModel]);

  useEffect(() => {
    if (!selectedModel) {
      setSelectedRun(null);
      return;
    }
    setSelectedRun((prev) => {
      const stillValid = prev && modelRuns.some((r) => r.run_id === prev.run_id);
      return stillValid ? prev : (modelRuns[0] ?? null);
    });
  }, [selectedModel?.model_id, modelRuns]);

  useEffect(() => {
    const runId = selectedRun?.run_id;
    if (!runId) {
      setRunLogs([]);
      setLogsError(null);
      setLoadingLogs(false);
      return;
    }

    let cancelled = false;
    setLoadingLogs(true);
    setLogsError(null);

    api
      .get(`/modelrun_logs/${encodeURIComponent(runId)}`)
      .then((res) => {
        if (!cancelled) setRunLogs(Array.isArray(res.data) ? res.data : []);
      })
      .catch((e) => {
        console.error("Failed to fetch run logs:", e);
        if (!cancelled) {
          setRunLogs([]);
          setLogsError("Failed to load logs for this run.");
        }
      })
      .finally(() => {
        if (!cancelled) setLoadingLogs(false);
      });

    return () => {
      cancelled = true;
    };
  }, [selectedRun?.run_id]);

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
    return [...runLogs].sort(
      (a, b) => new Date(b.timestamp_utc || 0) - new Date(a.timestamp_utc || 0)
    );
  }, [runLogs]);

  return (
    <Card className="flat-card dashboard-component model-details-panel">
      <Card.Body className="d-flex flex-column">
        <div className="model-panel__header">
          <div>
            <h5 className="model-panel__title mb-0">Model details</h5>
            <p className="model-panel__subtitle mb-0">
              {selectedModel ? selectedModel.name : "Waiting for model selection"}
            </p>
          </div>
        </div>

        {!selectedModel ? (
          <div className="model-details-empty">
            <FeatherIcon icon="arrow-left" size={48} className="model-details-empty__icon" />
            <div className="model-details-empty__title">No model selected</div>
            <p className="model-details-empty__text mb-0">
              Click a model in the list on the left to see its parameters, run history, and logs.
            </p>
          </div>
        ) : loading ? (
          <div className="text-center py-4">
            <Spinner animation="border" />
          </div>
        ) : (
          <div className="model-details-content">
            <section className="model-details-section model-details-section--params">
              <h6 className="model-details-section__title">Parameters</h6>
              <div className="model-details-params-wrap">
                {selectedModel.parameters ? (
                  <pre className="model-logs-params-pre mb-0">{JSON.stringify(selectedModel.parameters, null, 2)}</pre>
                ) : (
                  <div className="text-muted small">No parameters found.</div>
                )}
              </div>
            </section>

            <section className="model-details-section model-details-section--runs">
              <h6 className="model-details-section__title">Run history</h6>
              <div className="model-details-table-wrap model-details-table-wrap--runs">
                {modelRuns.length === 0 ? (
                  <div className="text-muted small model-details-scroll-empty">This model has no recorded runs.</div>
                ) : (
                  <Table size="sm" className="model-details-table mb-0">
                    <thead>
                      <tr>
                        <th>Run</th>
                        <th>Start</th>
                        <th>Finish</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {modelRuns.map((run) => (
                        <tr
                          key={run.run_id}
                          className={`cursor-pointer${
                            selectedRun?.run_id === run.run_id ? " model-details-table__row--selected" : ""
                          }`}
                          onClick={() => setSelectedRun(run)}
                        >
                          <td>{formatRunLabel(run)}</td>
                          <td>{formatDateTime(run.started_at)}</td>
                          <td>{formatDateTime(run.finished_at)}</td>
                          <td>{run.status || "N/A"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                )}
              </div>
            </section>

            <section className="model-details-section model-details-section--logs">
              <h6 className="model-details-section__title">Run logs</h6>
              <div className="model-details-table-wrap model-details-table-wrap--logs">
                {!selectedRun ? (
                  <div className="text-muted small model-details-scroll-empty">Select a run to view its logs.</div>
                ) : loadingLogs ? (
                  <div className="text-muted small model-details-scroll-empty">
                    <Spinner animation="border" size="sm" className="me-2" />
                    Loading…
                  </div>
                ) : logsError ? (
                  <div className="text-danger small model-details-scroll-empty">{logsError}</div>
                ) : sortedRunLogs.length === 0 ? (
                  <div className="text-muted small model-details-scroll-empty">No logs found for this run.</div>
                ) : (
                  <>
                    <Table size="sm" className="model-details-table mb-0">
                    <thead>
                      <tr>
                        <th>Timestamp</th>
                        <th>Sensor</th>
                        <th>Value</th>
                        <th>Message</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedRunLogs.slice(0, 200).map((row) => (
                        <tr key={row.inference_id ?? `${row.sensor_id}-${row.timestamp_utc}`}>
                          <td>{formatDateTime(row.timestamp_utc)}</td>
                          <td>{row.sensor_id ?? "—"}</td>
                          <td>{row.value ?? "—"}</td>
                          <td className="text-truncate">{row.inference_message ?? "—"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </Table>
                    {sortedRunLogs.length > 200 && (
                      <div className="text-muted small model-details-table__footer">Showing first 200 log rows.</div>
                    )}
                  </>
                )}
              </div>
            </section>
          </div>
        )}
      </Card.Body>
    </Card>
  );
}
