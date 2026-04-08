import React, { useEffect, useMemo, useState } from "react";
import { Card, Spinner, Table } from "react-bootstrap";
import monitoring_api from "../../../monitoring_api";

export default function HealthOverview({ refreshKey }) {
  const [components, setComponents] = useState([]);
  const [events, setEvents] = useState([]);
  const [loadingComponents, setLoadingComponents] = useState(true);
  const [loadingEvents, setLoadingEvents] = useState(true);

  const fetchComponents = async () => {
    try {
      const res = await monitoring_api.get("/components");
      setComponents(Array.isArray(res.data) ? res.data : []);
    } catch (error) {
      console.error("Failed to fetch components:", error);
      setComponents([]);
    }
    setLoadingComponents(false);
  };

  const fetchEvents = async () => {
    try {
      const res = await monitoring_api.get("/events");
      setEvents(Array.isArray(res.data) ? res.data : []);
    } catch (error) {
      console.error("Failed to fetch events:", error);
      setEvents([]);
    }
    setLoadingEvents(false);
  };

  useEffect(() => {
    setLoadingComponents(true);
    setLoadingEvents(true);
    fetchComponents();
    fetchEvents();
  }, [refreshKey]);

  const { healthPercent, activeCount, totalCount } = useMemo(() => {
    const total = components.length;
    const active = components.filter((c) => c.status === "active").length;
    const pct = total === 0 ? 0 : Math.round((active / total) * 100);
    return { healthPercent: pct, activeCount: active, totalCount: total };
  }, [components]);

  const errorEvents = useMemo(() => {
    return events
      .filter((e) => e.severity === "ERROR" || e.severity === "CRITICAL")
      .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
  }, [events]);

  return (
    <Card className="flat-card">
      <Card.Body>
        <div className="border-bottom d-flex justify-content-between align-items-center mb-2">
          <h3 className="mb-0" style={{ fontSize: "1.1rem" }}>
            Health overview
          </h3>
        </div>

        <div className="mt-2">
          <div className="fw-semibold">System health</div>
          {loadingComponents ? (
            <div className="text-muted small mt-1">
              <Spinner animation="border" size="sm" className="me-2" />
              Loading…
            </div>
          ) : (
            <div className="mt-1">
              <div className="fw-bold text-primary" style={{ fontSize: "2rem" }}>
                {healthPercent}%
              </div>
              <div className="text-muted small">
                {activeCount} / {totalCount} components active
              </div>
            </div>
          )}
        </div>

        <div className="mt-3">
          <div className="fw-semibold" style={{ fontSize: "0.9rem" }}>
            Error events{" "}
            {!loadingEvents ? (
              <span className="text-muted">({errorEvents.length})</span>
            ) : null}
          </div>

          {loadingEvents ? (
            <div className="text-muted small mt-1">
              <Spinner animation="border" size="sm" className="me-2" />
              Loading…
            </div>
          ) : errorEvents.length === 0 ? (
            <div className="text-muted small mt-1">No ERROR/CRITICAL events</div>
          ) : (
            <div className="events-scroll mt-2" >
              <Table striped bordered hover responsive size="sm" className="mb-0">
                <thead>
                  <tr>
                    <th>Severity</th>
                    <th>Component</th>
                    <th>Message</th>
                  </tr>
                </thead>
                <tbody>
                  {errorEvents.slice(0, 5).map((e) => (
                    <tr key={e.event_id}>
                      <td>
                        <span
                          className={`badge ${
                            e.severity === "CRITICAL" ? "bg-dark" : "bg-danger"
                          }`}
                        >
                          {e.severity}
                        </span>
                      </td>
                      <td>
                        {e.component_name} ({e.component_instance_id})
                      </td>
                      <td>{e.message}</td>
                    </tr>
                  ))}
                </tbody>
              </Table>
            </div>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}