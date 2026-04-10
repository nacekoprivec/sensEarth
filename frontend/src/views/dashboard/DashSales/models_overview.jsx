import React, { useEffect, useMemo, useState } from "react";
import { Card, Spinner } from "react-bootstrap";
import api from "../../../api";
import monitoring_api from "../../../monitoring_api";

function ModelRun({ completed, ongoing }) {
  const total = completed + ongoing;
  const successRate = total > 0 ? (completed / total) * 100 : null;

  return (
    <div>
      <div style={{fontSize: "1.2rem", fontWeight: 600, fontVariantNumeric: "tabular-nums",}}>
        {completed} / {total}
      </div>

      <div className="text-muted" style={{ fontSize: "0.7rem" }}>
        Completed · Ongoing: {ongoing}
      </div>

      {successRate != null && (
        <div
          style={{
            fontSize: "0.7rem",
            color: successRate > 90 ? "#15803d" : "#b45309",
          }}
        >
          {successRate.toFixed(0)}% success
        </div>
      )}
    </div>
  );
}

function MiniCard({ title, children }) {
  return (
    <div
      style={{
        flex: 1,
        minWidth: "120px",
        padding: "8px 10px",
        border: "1px solid #e9ecef",
        borderRadius: "10px",
        background: "#fff",
      }}
    >
      <div className="fw-semibold text-muted" style={{ fontSize: "0.7rem" }}>
        {title}
      </div>
      <div className="mt-1">{children}</div>
    </div>
  );
}

export default function ModelsOverview({ refreshKey }) {
  const [models, setModels] = useState([]);
  const [modelRuns, setModelRuns] = useState([]);
  const [events, setEvents] = useState([]);

  const [loadingModels, setLoadingModels] = useState(true);
  const [loadingRuns, setLoadingRuns] = useState(true);
  const [loadingEvents, setLoadingEvents] = useState(true);

  useEffect(() => {
    setLoadingModels(true);
    setLoadingRuns(true);
    setLoadingEvents(true);

    api.get("/models")
      .then(res => setModels(Array.isArray(res.data) ? res.data : []))
      .catch(() => setModels([]))
      .finally(() => setLoadingModels(false));

    api.get("/modelRuns")
      .then(res => setModelRuns(Array.isArray(res.data) ? res.data : []))
      .catch(() => setModelRuns([]))
      .finally(() => setLoadingRuns(false));

    monitoring_api.get("/events")
      .then(res => setEvents(Array.isArray(res.data) ? res.data : []))
      .catch(() => setEvents([]))
      .finally(() => setLoadingEvents(false));

  }, [refreshKey]);

  const { completedRuns, ongoingRuns } = useMemo(() => {
    const completed = modelRuns.filter(
      r => r.status === "completed" || r.finished_at
    ).length;

    return {
      completedRuns: completed,
      ongoingRuns: Math.max(0, modelRuns.length - completed),
    };
  }, [modelRuns]);

  const modelIssues = useMemo(() => {
    const errEvents = events.filter(
      e => e.severity === "ERROR" || e.severity === "CRITICAL" || e.severity === "WARNING"
    );

    const counts = new Map();
    models.forEach(m => counts.set(m.name, 0));

    errEvents.forEach(e => {
      const name = e?.metadata?.model_name;
      if (name && counts.has(name)) {
        counts.set(name, counts.get(name) + 1);
      }
    });

    return Array.from(counts.entries())
      .map(([modelName, issues]) => ({ modelName, issues }))
      .sort((a, b) => b.issues - a.issues);
  }, [events, models]);

  const anyIssues = modelIssues.some(x => x.issues > 0);

  return (
    <Card className="flat-card">
      <Card.Body>
        <div className="border-bottom mb-2">
          <h3 style={{ fontSize: "1rem", fontWeight: 600 }}>
            Model runs
          </h3>
        </div>

        <div className="d-flex flex-wrap gap-2">

          {/* Runs */}
          <MiniCard title="Runs">
            {loadingRuns ? (
              <Spinner size="sm" />
            ) : (
              <ModelRun
                completed={completedRuns}
                ongoing={ongoingRuns}
              />
            )}
          </MiniCard>

          {/* Models */}
          <MiniCard title="Models">
            {loadingModels ? (
              <Spinner size="sm" />
            ) : (
              <div
                style={{
                  fontSize: "1.2rem",
                  fontWeight: 600,
                  color: "#1d4ed8",
                }}
              >
                {models.length}
              </div>
            )}
          </MiniCard>

          {/* Issues */}
          <MiniCard title="Issues">
            {loadingEvents || loadingModels ? (
              <Spinner size="sm" />
            ) : !anyIssues ? (
              <div className="text-muted" style={{ fontSize: "0.75rem" }}>
                None
              </div>
            ) : (
              <div style={{ fontSize: "0.75rem" }}>
                {modelIssues
                  .filter(x => x.issues > 0)
                  .slice(0, 3)
                  .map(x => (
                    <div
                      key={x.modelName}
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                      }}
                    >
                      <span style={{ maxWidth: "100px" }}>
                        {x.modelName}
                      </span>
                      <span
                        style={{
                          color: "#b91c1c",
                          fontWeight: 600,
                        }}
                      >
                        {x.issues}
                      </span>
                    </div>
                  ))}
              </div>
            )}
          </MiniCard>

        </div>
      </Card.Body>
    </Card>
  );
}