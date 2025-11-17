import { useEffect, useState } from "react";
import { Row, Col, Card, Accordion, Button, Spinner } from "react-bootstrap";

export default function BasicTypography() {
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);

  // Fetch logs from backend
  const fetchLogs = async () => {
    setLoading(true);
    const res = await fetch("http://localhost:8000/logs"); // Adjust URL
    const data = await res.json();
    setLogs(data);
    setLoading(false);
  };

  useEffect(() => {
    fetchLogs();
  }, []);

  // Delete log
  const deleteLog = async (id) => {
    if (!window.confirm("Delete this log?")) return;
    await fetch(`http://localhost:8000/logs/${id}`, { method: "DELETE" });
    fetchLogs(); // refresh list
  };

  return (
    <Row>
      <Col sm={12}>
        <Card>
          <Card.Header>
            <Card.Title as="h5">Logs</Card.Title>
          </Card.Header>
          <Card.Body>
            {loading ? (
              <Spinner animation="border" />
            ) : (
              <Accordion alwaysOpen>
                {logs.map((log) => (
                  <Accordion.Item key={log.id} eventKey={String(log.id)}>
                    <Accordion.Header>
                      <div className="d-flex justify-content-between w-100 align-items-center">
                        <span>
                          <strong>Log {log.id}</strong> | {log.start_timedate} →{" "}
                          {log.end_timedate} | Duration  →{" "} {log.duration_formated} 
                        </span>
                        <Button
                          variant="danger"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation(); // prevent accordion toggle
                            deleteLog(log.id);
                          }}
                        >
                        </Button>
                      </div>
                    </Accordion.Header>
                   <Accordion.Body>
                      <pre style={{ whiteSpace: "pre-wrap" }}>
                        <h6>Configuration</h6>
                        <div style={{
                          border: "1px solid #ddd",
                          padding: "8px",
                          borderRadius: "5px",
                          background: "#f9f9f9",
                        }}>
                          {JSON.stringify(log.config, null, 2)}
                        </div>
                        <div style={{
                          border: "1px solid #ddd",
                          padding: "8px",
                          borderRadius: "5px",
                          background: "#f9f9f9",
                        }}>
                        Precision: {log.precision} | Recall: {log.recall} | F1: {log.f1}
                        </div>
                      </pre>
                      <h6>Anomalies</h6>
                      <div
                        style={{
                          maxHeight: "200px",
                          overflowY: "auto",
                          border: "1px solid #ddd",
                          padding: "8px",
                          borderRadius: "5px",
                          background: "#f9f9f9",
                        }}
                      >
                        {log.anomalies.length > 0 ? (
                          log.anomalies.map((a) => (
                            <div key={a.id}>
                              <strong>t={a.timestamp}:</strong>{" "}
                              {a.ftr_vector}
                            </div>
                          ))
                        ) : (
                          <i>No anomalies</i>
                        )}
                      </div>
                    </Accordion.Body>
                  </Accordion.Item>
                ))}
              </Accordion>
            )}
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );
}
