import React, { useEffect, useMemo, useState } from "react";
import { Card, Spinner, Table } from "react-bootstrap";
import { PieChart, Pie, Cell } from "recharts";

import monitoring_api from "../../monitoring_api";

function PieCircle({ percent }) {
  const data = [
    { name: "filled", value: percent },
    { name: "remaining", value: 100 - percent },
  ];

  const COLORS = ["#7267ef", "#e9ecef"];

  return (
    <PieChart width={56} height={56}>
      <Pie
        data={data}
        innerRadius={20}
        outerRadius={28}
        startAngle={90}
        endAngle={-270}
        dataKey="value"
      >
        {data.map((_, index) => (
          <Cell key={index} fill={COLORS[index]} />
        ))}
      </Pie>
    </PieChart>
  );
}

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

  const severityCounts = useMemo(() => {
    return events.reduce(
      (acc, e) => {
        acc[e.severity] = (acc[e.severity] || 0) + 1;
        return acc;
      },
      { CRITICAL: 0, ERROR: 0, WARNING: 0 }
    );
  }, [events]);

  return (
    <Card className="flat-card">
      <Card.Body>
        <div className="border-bottom d-flex align-items-center mb-2">
          <h3 className="mb-0" style={{ fontSize: "1.1rem" }}>
            Health overview
          </h3>
        </div>

        {/* System health */}
        <div className="mt-2">

          {loadingComponents ? (
            <div className="text-muted small mt-1">
              <Spinner animation="border" size="sm" className="me-2" />
              Loading…
            </div>
          ) : (
            <div className="health-overview-panel mt-1">
              <div className="health-overview-main border rounded-3 p-2 shadow-sm d-flex flex-wrap align-items-center justify-content-center gap-3">
                <div className="d-flex flex-column justify-content-center text-center">
                  <div
                    className="fw-bold text-primary health-overview-percent"
                  >
                    {healthPercent}%
                  </div>
                  <div className="text-muted small mt-1">
                    {activeCount} / {totalCount} components active
                  </div>
                </div>

                <div className="d-flex align-items-center justify-content-center flex-shrink-0">
                  <div style={{ width: 56, height: 56 }}>
                    <PieCircle percent={healthPercent} />
                  </div>
                </div>
              </div>

              {!loadingEvents && (
                <div className="health-overview-severity border rounded-3 p-2 shadow-sm bg-white">
                  <div className="d-flex justify-content-between align-items-center px-2 py-2 border-bottom gap-2">
                    <div className="bg-dark text-white px-2 py-1 fw-semibold rounded small text-center text-nowrap flex-shrink-0">
                      CRITICAL
                    </div>
                    <div className="fw-bold fs-6">
                      {severityCounts.CRITICAL}
                    </div>
                  </div>

                  <div className="d-flex justify-content-between align-items-center px-2 py-2 border-bottom gap-2">
                    <div className="bg-danger text-white px-2 py-1 fw-semibold rounded small text-center text-nowrap flex-shrink-0">
                      ERROR
                    </div>
                    <div className="fw-bold fs-6">
                      {severityCounts.ERROR}
                    </div>
                  </div>

                  <div className="d-flex justify-content-between align-items-center px-2 py-2 gap-2">
                    <div className="bg-warning text-dark px-2 py-1 fw-semibold rounded small text-center text-nowrap flex-shrink-0">
                      WARNING
                    </div>
                    <div className="fw-bold fs-6">
                      {severityCounts.WARNING}
                    </div>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}