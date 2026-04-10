import React, { useState, useEffect } from "react";
import { Card, Spinner, Badge } from "react-bootstrap";
import monitoring_api from "../../../monitoring_api";

export default function IngestionStatus({ modelsUpdated }) {
  const [activeScraperCount, setActiveScraperCount] = useState(0);
  const [totalScraperCount, setTotalScraperCount] = useState(0);
  const [scraperErrorEvents, setScraperErrorEvents] = useState([]);
  const [loadingComponents, setLoadingComponents] = useState(true);
  const [loadingEvents, setLoadingEvents] = useState(true);

  const fetchScraperCounts = async () => {
    try {
      const res = await monitoring_api.get("/components");
      const list = Array.isArray(res.data) ? res.data : [];
      const scrapers = list.filter((c) => c.type === "scraper");

      setTotalScraperCount(scrapers.length);
      setActiveScraperCount(
        scrapers.filter((c) => c.status === "active").length
      );
    } catch (error) {
      console.error("Failed to fetch components:", error);
      setActiveScraperCount(0);
      setTotalScraperCount(0);
    }
    setLoadingComponents(false);
  };

  const fetchScraperErrorEvents = async () => {
    try {
      const res = await monitoring_api.get("/events");
      const list = Array.isArray(res.data) ? res.data : [];

      const errs = list
        .filter(
          (e) =>
            e.component_name === "scraper" &&
            (e.severity === "ERROR" || e.severity === "CRITICAL" || e.severity === "WARNING")
        )
        .sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));

      setScraperErrorEvents(errs);
    } catch (error) {
      console.error("Failed to fetch events:", error);
      setScraperErrorEvents([]);
    }
    setLoadingEvents(false);
  };

  useEffect(() => {
    setLoadingComponents(true);
    setLoadingEvents(true);
    fetchScraperCounts();
    fetchScraperErrorEvents();
  }, [modelsUpdated]);

  const healthyCount = activeScraperCount;
  const downCount = totalScraperCount - activeScraperCount;
  const degradedCount = 0;

  const totalErrors = scraperErrorEvents.length;
  const criticalCount = scraperErrorEvents.filter(
    (e) => e.severity === "CRITICAL"
  ).length;
  const warningCount = scraperErrorEvents.filter(
    (e) => e.severity === "WARNING"
  ).length;

  const safePercent = (value) =>
    totalScraperCount ? (value / totalScraperCount) * 100 : 0;

  return (
    <Card className="flat-card">
      <Card.Body className="d-flex flex-column gap-3">
        {/* Header */}
        <div className="d-flex justify-content-between align-items-center border-bottom pb-2">
          <h5 className="mb-0 fw-semibold">Scraper overview</h5>
        </div>

        {/* Loading */}
        {loadingComponents ? (
          <div className="d-flex justify-content-center align-items-center py-4">
            <Spinner animation="border" size="sm" />
          </div>
        ) : (
          <>
            {/* Status bar */}
            <div>
              <div
                className="d-flex w-100 overflow-hidden"
                style={{ height: "10px", borderRadius: "6px" }}
              >
                <div
                  style={{
                    width: `${safePercent(healthyCount)}%`,
                    backgroundColor: "#198754",
                  }}
                />
                <div
                  style={{
                    width: `${safePercent(degradedCount)}%`,
                    backgroundColor: "#ffc107",
                  }}
                />
                <div
                  style={{
                    width: `${safePercent(downCount)}%`,
                    backgroundColor: "#dc3545",
                  }}
                />
              </div>

              {/* Total */}
              <div className="d-flex justify-content-center mt-2 small text-muted">
                <span className="fw-semibold text-dark">
                  Total scrapers: {totalScraperCount}
                </span>
              </div>
            </div>

            {/* Summary cards */}
            <div className="d-flex justify-content-between text-center gap-2">
              <div className="flex-fill p-2 rounded bg-light">
                <div className="text-success fw-semibold small">Healthy</div>
                <div className="fs-6 fw-bold">{healthyCount}</div>
              </div>

              <div className="flex-fill p-2 rounded bg-light">
                <div className="text-warning fw-semibold small">Degraded</div>
                <div className="fs-6 fw-bold">{degradedCount}</div>
              </div>

              <div className="flex-fill p-2 rounded bg-light">
                <div className="text-danger fw-semibold small">Down</div>
                <div className="fs-6 fw-bold">{downCount}</div>
              </div>
            </div>

            <div className="d-flex justify-content-between align-items-center border-top pt-2">
              
              {/* Warning indicator */}
              <div className="d-flex gap-3 align-items-center mt-2">
                <span className="small text-muted">Warnings</span>
                <Badge bg={warningCount > 0 ? "warning" : "secondary"}>
                  {warningCount}
                </Badge>
              </div>

              {/* Error indicator */}
              <div className="d-flex gap-3 align-items-center mt-2">
                <span className="small text-muted">Errors</span>
                <Badge bg={totalErrors > 0 ? "danger" : "secondary"}>
                  {totalErrors}
                </Badge>
              </div>

              {/* Critical indicator */}
              <div className="d-flex gap-3 align-items-center mt-2">
                <span className="small text-muted">Critical</span>
                <Badge bg={criticalCount > 0 ? "dark" : "secondary"}>
                  {criticalCount}
                </Badge>
              </div>
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );
}