import React, { useState, useEffect, useMemo } from "react";
import { Card, Table, Spinner, Accordion, Form, InputGroup, Button } from "react-bootstrap";
import FeatherIcon from "feather-icons-react";
import monitoring_api from "../../monitoring_api";

const LEVELS = ["INFO", "WARN", "ERROR", "CRITICAL"];

export default function EventsDashboard() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await monitoring_api.get("/events");
        setEvents(Array.isArray(res.data) ? res.data : []);
      } catch (error) {
        console.error("Failed to fetch events:", error);
        setEvents([]);
      }
      setLoading(false);
    })();
  }, []);

  const filteredEvents = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return events;
    return events.filter((e) =>
      `${e.component_name || ""} ${e.component_instance_id || ""} ${e.event_type || ""} ${e.message || ""}`
        .toLowerCase()
        .includes(query)
    );
  }, [events, searchQuery]);

  return (
    <Card className="flat-card dashboard-component monitoring-dashboard events-dashboard">
      <Card.Body>
        <div className="monitoring-dashboard__header">
          <div className="monitoring-dashboard__title-row">
            <div>
              <h5 className="monitoring-dashboard__title mb-0">Monitoring Events</h5>
              <p className="monitoring-dashboard__subtitle mb-0">
                {loading ? "Loading events..." : `${filteredEvents.length} of ${events.length} shown`}
              </p>
            </div>
            <InputGroup size="sm" className="monitoring-dashboard__search">
              <InputGroup.Text>
                <FeatherIcon icon="search" size={14} />
              </InputGroup.Text>
              <Form.Control
                type="search"
                placeholder="Search events..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search events"
              />
              {searchQuery && (
                <Button variant="outline-secondary" size="sm" onClick={() => setSearchQuery("")} aria-label="Clear search">
                  <FeatherIcon icon="x" size={14} />
                </Button>
              )}
            </InputGroup>
          </div>
        </div>

        <div className="monitoring-dashboard__table-wrap events-dashboard__accordion-wrap">
          {loading ? (
            <div className="monitoring-dashboard__loading">
              <Spinner animation="border" />
            </div>
          ) : (
            <Accordion defaultActiveKey="0" flush>
              {LEVELS.map((level, index) => {
                const levelEvents = filteredEvents.filter((e) => e.severity === level);
                return (
                  <Accordion.Item eventKey={index.toString()} key={level}>
                    <Accordion.Header>
                      <span className={`events-severity-badge events-severity-badge--${level.toLowerCase()}`}>{level}</span>
                      <span className="events-count-badge">{levelEvents.length}</span>
                    </Accordion.Header>
                    <Accordion.Body className="p-0">
                      <div className="monitoring-dashboard__table-wrap events-dashboard__panel">
                        <Table className="dashboard-table monitoring-table events-table mb-0" hover>
                          <thead>
                            <tr>
                              <th>Component</th>
                              <th>Type</th>
                              <th>Message</th>
                              <th>Timestamp</th>
                            </tr>
                          </thead>
                          <tbody>
                            {levelEvents.length === 0 ? (
                              <tr>
                                <td colSpan="4" className="text-center monitoring-table__empty">
                                  {events.length === 0 ? "No events found" : `No ${level} events match your search`}
                                </td>
                              </tr>
                            ) : (
                              levelEvents.map((e) => (
                                <tr key={e.event_id}>
                                  <td className="events-table__component">
                                    <div className="monitoring-table__name">{e.component_name}</div>
                                    <div className="monitoring-table__instance">{e.component_instance_id}</div>
                                  </td>
                                  <td>{e.event_type}</td>
                                  <td>{e.message}</td>
                                  <td>{new Date(e.timestamp).toLocaleString()}</td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </Table>
                      </div>
                    </Accordion.Body>
                  </Accordion.Item>
                );
              })}
            </Accordion>
          )}
        </div>
      </Card.Body>
    </Card>
  );
}
