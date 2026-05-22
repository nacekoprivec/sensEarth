import React, { useState, useEffect } from "react";
import { Row, Col, Card, Table, Spinner, Accordion } from "react-bootstrap";
import monitoring_api from "../../monitoring_api";

export default function EventsDashboard() {
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchEvents = async () => {
    try {
      const res = await monitoring_api.get("/events");
      const eventsData = Array.isArray(res.data) ? res.data : [];
      if (!Array.isArray(res.data)) {
        console.warn("Monitoring events response was not an array, defaulting to []:", res.data);
      }
      setEvents(eventsData);
      console.log("Fetched events:", eventsData);
    } catch (error) {
      console.error("Failed to fetch events:", error);
      setEvents([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  const levels = ["INFO", "WARN", "ERROR", "CRITICAL"];

  const groupedEvents = (level) =>
    events.filter((e) => e.severity === level);

  const badgeColor = (level) => {
    switch (level) {
      case "INFO":
        return "bg-info";
      case "WARN":
        return "bg-warning";
      case "ERROR":
        return "bg-danger";
      case "CRITICAL":
        return "bg-dark";
      default:
        return "bg-secondary";
    }
  };

  return (
    <Card className="flat-card  events-scroll dashboard-component">
      <Card.Body>
        <div className="border-bottom d-flex justify-content-between align-items-center mb-3" > 
                <h3 className="mb-3">Monitoring Events</h3>
              </div>
            {loading ? (
              <div className="text-center">
                <Spinner animation="border" />
              </div>
            ) : (
              <Accordion>
                {levels.map((level, index) => {
                  const filtered = groupedEvents(level);

                  return (
                    <Accordion.Item eventKey={index.toString()} key={level}>
                      <Accordion.Header>
                        <span className={`badge me-2 ${badgeColor(level)}`}>
                          {level}
                        </span>
                        {level} ({filtered.length})
                      </Accordion.Header>

                      <Accordion.Body>
                        <Table striped bordered hover responsive>
                          <thead>
                            <tr>
                              <th>Component</th>
                              <th>Type</th>
                              <th>Message</th>
                              <th>Timestamp</th>
                            </tr>
                          </thead>
                          <tbody>
                            {filtered.length === 0 ? (
                              <tr>
                                <td colSpan="4" className="text-center">
                                  No {level} events
                                </td>
                              </tr>
                            ) : (
                              filtered.map((e) => (
                                <tr key={e.event_id}>
                                  <td>
                                    {e.component_name} (
                                    {e.component_instance_id})
                                  </td>
                                  <td>{e.event_type}</td>
                                  <td>{e.message}</td>
                                  <td>
                                    {new Date(e.timestamp).toLocaleString()}
                                  </td>
                                </tr>
                              ))
                            )}
                          </tbody>
                        </Table>
                      </Accordion.Body>
                    </Accordion.Item>
                  );
                })}
              </Accordion>
            )}
          </Card.Body>
        </Card>
  );
}