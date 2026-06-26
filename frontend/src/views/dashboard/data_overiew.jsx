import React, { useEffect, useState } from "react";
import { Card, Spinner } from "react-bootstrap";
import monitoring_api from "../../monitoring_api";
import api from '../../api';

function StatCard({ label, value, subtext, variant = "default" }) {
  const colors = {
    default: {
      bg: "#ffffff",
      border: "#e9ecef",
      value: "#212529",
    },
    success: {
      bg: "#f0fdf4",
      border: "#bbf7d0",
      value: "#15803d",
    },
    danger: {
      bg: "#fef2f2",
      border: "#fecaca",
      value: "#b91c1c",
    },
    warning: {
      bg: "#fffbeb",
      border: "#fde68a",
      value: "#b45309",
    },
    info: {
      bg: "#eff6ff",
      border: "#bfdbfe",
      value: "#1d4ed8",
    },
  };

  const c = colors[variant];

  return (
    <div
      className="data-overview-stat"
      style={{
        background: c.bg,
        border: `1px solid ${c.border}`,
      }}
    >
      <div className="data-overview-stat__label">{label}</div>

      <div className="data-overview-stat__value" style={{ color: c.value }}>
        {value}
      </div>

      {subtext && (
        <div className="data-overview-stat__sub">{subtext}</div>
      )}
    </div>
  );
}

export default function DataOverview({ refreshKey }) {
  const [loading, setLoading] = useState(true);
  const [data, setData] = useState({});
  
  const fetchSensorsAll = async () => {
    try {
      const res = await api.get("/sensors");
      console.log("Fetched all sensors for data overview:", res.data);
      const activeSensors = res.data.length;

      setData((d) => ({ ...d, activeSensors }));

    } catch (error) {
      console.error("Failed to fetch sensors for data overview:", error);
    }
  };


  const fetchStructuredStorage = async () => {
    try {
      const res = await monitoring_api.get("/events");
      console.log("Fetched events for data overview:", res.data);
      const list = Array.isArray(res.data) ? res.data : [];

      const metricsRes = await monitoring_api.get("/metrics");
      const metrics = Array.isArray(metricsRes.data) ? metricsRes.data : [];

      const middlewareEvents = list
        .filter(
          (e) =>
            e.component_name === "middleware" &&
            e.event_type === "data_ingest_completed" &&
            typeof e.message === "string"
        )
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

      const scraperEvents = list
        .filter(
          (e) =>
            e.component_name === "scraper" &&
            typeof e.message === "string"
        )
        .sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));

      const latest = middlewareEvents[middlewareEvents.length - 1];
      const latestScraper = scraperEvents[scraperEvents.length - 1];

      if (!latest) {
        setData({});
      } else {
        const counts = middlewareEvents.map((e) => {
          const m = e.message.match(/Inserted\s+(\d+)\s+measurements/i);
          return m ? Number(m[1]) : 0;
        });

        const total = counts.reduce((a, b) => a + b, 0);

        const latestMatch = latest.message.match(/Inserted\s+(\d+)\s+measurements/i);
        const latestCount = latestMatch ? Number(latestMatch[1]) : null;

        // Extract duplicates count from metrics
        const dupMetrics = metrics.filter(m => m.metric_name === "duplicates_skipped");
        const numDuplicates = dupMetrics.reduce((sum, m) => sum + m.value, 0) || null;

        // Extract invalid measurements % from metrics
        const invalidMetric = metrics.find(m => m.metric_name === "measurements_skipped_rate");
        const invalidPercent = invalidMetric ? invalidMetric.value : null;

        // Extract failed ingestions from middleware messages
        const failedMatch = latest.message.match(/(\d+)\s+failed/i);
        const failedCount = failedMatch ? Number(failedMatch[1]) : 0;

        // Calculate success rate
        let successRate = null;
        if (latestCount !== null && failedCount >= 0) {
          const totalAttempts = latestCount + failedCount;
          if (totalAttempts > 0) {
            successRate = (latestCount / totalAttempts) * 100;
          }
        }

        // Average ingested per calendar day that had at least one ingest event
        const ingestDays = new Set(
          middlewareEvents.map((e) => new Date(e.timestamp).toISOString().slice(0, 10))
        ).size;

        let ratePerDay = null;
        if (ingestDays > 0) ratePerDay = total / ingestDays;

        setData({
          total,
          latestCount,
          ratePerDay,
          ingestDays,
          numDuplicates,
          invalidPercent,
          failedCount,
          successRate,
          lastTimestamp: latest.timestamp,
        });
      }
    } catch (e) {
      console.error(e);
      setData({});
    }
  };

    useEffect(() => {
    const load = async () => {
      setLoading(true);

      try {
        await Promise.all([
          fetchSensorsAll(),
          fetchStructuredStorage()
        ]);
      } catch (e) {
        console.error(e);
      }

      setLoading(false);
    };

    load();
  }, [refreshKey]);

  return (
    <Card className="flat-card">
      <Card.Body>
        <div className="border-bottom d-flex align-items-center mb-2">
          <h3 className="mb-0" style={{ fontSize: "1.1rem" }}>Data overview</h3>
        </div>

        {loading ? (
          <div className="text-muted small">
            <Spinner animation="border" size="sm" className="me-2" />
            Loading…
          </div>
        ) : (
          <>
            <div className="data-overview-stats">
                <StatCard
                  label="Duplicates"
                  value={
                    data.numDuplicates == null
                      ? "0"
                      : `${data.numDuplicates}`
                  }
                />

                <StatCard
                  label="Invalid measurements"
                  value={
                    data.invalidPercent == null
                      ? "—"
                      : `${data.invalidPercent.toLocaleString(undefined, {
                        maximumFractionDigits: 1,
                      })}%`
                  }
                />

                <StatCard
                  label="Failed ingestions"
                  value={
                    data.failedCount == null
                      ? "—"
                      : `${data.failedCount}`
                  }
                />

                <StatCard
                  label="Active sensors"
                  value={
                    data.activeSensors == null
                      ? "—"
                      : `${data.activeSensors}`
                  }
                />
                <StatCard
                  label="Total ingested"
                  value={
                    data.total == null
                      ? "—"
                      : `${data.total.toLocaleString()} records`
                  }
                  subtext={
                    data.ingestDays == null
                      ? null
                      : `${data.ingestDays} day${data.ingestDays === 1 ? "" : "s"}`
                  }
                />

                <StatCard
                  label="Last batch"
                  value={
                    data.latestCount == null
                      ? "—"
                      : `${data.latestCount} records`
                  }
                />

                <StatCard
                  label="Ingestion rate"
                  value={
                    data.ratePerDay == null
                      ? "—"
                      : `${data.ratePerDay.toLocaleString(undefined, {
                        maximumFractionDigits: 1,
                      })}/d`
                  }
                />
                <StatCard
                  label="Success rate"
                  value={
                    data.successRate == null
                      ? "—"
                      : `${data.successRate.toLocaleString(undefined, {
                        maximumFractionDigits: 1,
                      })}%`
                  }
                />
            </div>

            <div className="mt-3 text-muted" style={{ fontSize: "0.8rem" }}>
              {data.lastTimestamp
                ? `Last ingestion: ${new Date(
                  data.lastTimestamp
                ).toLocaleString()}`
                : "No recent ingestion event found"}
            </div>
          </>
        )}
      </Card.Body>
    </Card>
  );
}