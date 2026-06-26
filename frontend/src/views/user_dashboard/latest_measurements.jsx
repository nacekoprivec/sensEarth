import React, { useMemo, useState } from "react";
import { Card, Table, Spinner, Form, InputGroup, Button } from "react-bootstrap";
import FeatherIcon from "feather-icons-react";

function formatRelativeTime(value) {
  if (!value) return "—";
  const diff = Date.now() - new Date(value).getTime();
  const sec = Math.floor(diff / 1000);
  if (sec < 60) return "just now";
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min} min ago`;
  const hrs = Math.floor(min / 60);
  if (hrs < 24) return `${hrs} hr ago`;
  return new Date(value).toLocaleString();
}

function formatLocation(locationStr) {
  if (!locationStr) return "—";
  try {
    const loc = JSON.parse(locationStr);
    if (!loc.coordinates || loc.coordinates.length < 2) return "—";
    const [lon, lat] = loc.coordinates;
    return (
      <a
        className="measurements-table__location"
        href={`https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}`}
        target="_blank"
        rel="noopener noreferrer"
      >
        {lat.toFixed(5)}, {lon.toFixed(5)}
      </a>
    );
  } catch {
    return "—";
  }
}

export default function LatestMeasurementsDashboard({ sensors, loading, onRefresh }) {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredSensors = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return sensors;
    return sensors.filter((s) =>
      `${s.sensor_label || ""} ${s.sensor_id || ""}`.toLowerCase().includes(query)
    );
  }, [sensors, searchQuery]);

  return (
    <Card className="flat-card dashboard-component monitoring-dashboard measurements-dashboard">
      <Card.Body>
        <div className="monitoring-dashboard__header">
          <div className="monitoring-dashboard__title-row">
            <div>
              <h5 className="monitoring-dashboard__title mb-0">Sensor Measurements</h5>
              <p className="monitoring-dashboard__subtitle mb-0">
                {loading ? "Loading measurements..." : `${filteredSensors.length} of ${sensors.length} shown`}
              </p>
            </div>
            <div className="measurements-dashboard__controls">
              <InputGroup size="sm" className="monitoring-dashboard__search">
                <InputGroup.Text>
                  <FeatherIcon icon="search" size={14} />
                </InputGroup.Text>
                <Form.Control
                  type="search"
                  placeholder="Search sensor or ID..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  aria-label="Search sensor or ID"
                />
                {searchQuery && (
                  <Button variant="outline-secondary" size="sm" onClick={() => setSearchQuery("")} aria-label="Clear search">
                    <FeatherIcon icon="x" size={14} />
                  </Button>
                )}
              </InputGroup>
              {onRefresh && (
                <Button variant="outline-secondary" size="sm" className="measurements-dashboard__refresh" onClick={onRefresh} aria-label="Refresh">
                  <FeatherIcon icon="refresh-cw" size={14} />
                </Button>
              )}
            </div>
          </div>
        </div>

        <div className="monitoring-dashboard__table-wrap">
          {loading ? (
            <div className="monitoring-dashboard__loading">
              <Spinner animation="border" />
            </div>
          ) : (
            <Table className="dashboard-table monitoring-table measurements-table mb-0" hover>
              <colgroup>
                <col className="measurements-table__col-sensor" />
                <col className="measurements-table__col-id" />
                <col className="measurements-table__col-time" />
                <col className="measurements-table__col-location" />
                <col className="measurements-table__col-value" />
              </colgroup>
              <thead>
                <tr>
                  <th>Sensor</th>
                  <th>ID</th>
                  <th>Time</th>
                  <th>Location</th>
                  <th className="text-end">Value</th>
                </tr>
              </thead>
              <tbody>
                {filteredSensors.length === 0 ? (
                  <tr>
                    <td colSpan="5" className="text-center monitoring-table__empty">
                      {sensors.length === 0 ? "No measurements found" : "No measurements match your search"}
                    </td>
                  </tr>
                ) : (
                  filteredSensors.map((sensor, index) => (
                    <tr key={`${sensor.sensor_id}-${sensor.timestamp_utc}-${index}`}>
                      <td className="measurements-table__sensor">{sensor.sensor_label}</td>
                      <td className="measurements-table__id">{sensor.sensor_id}</td>
                      <td className="measurements-table__time">{formatRelativeTime(sensor.timestamp_utc)}</td>
                      <td>{formatLocation(sensor.location)}</td>
                      <td>
                        <div className="measurements-table__value">
                          <span>{sensor.value}</span>
                          <FeatherIcon icon="bar-chart-2" size={14} className="measurements-table__value-icon" />
                        </div>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </Table>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}
