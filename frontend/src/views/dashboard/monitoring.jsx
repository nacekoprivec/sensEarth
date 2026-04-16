import React, { useState, useEffect } from "react";
import { Row, Col, Card, Table, Spinner } from "react-bootstrap";
import monitoring_api from "../../monitoring_api"; 

export default function MonitoringDashboard( {modelsUpdated} ) {
  const [components, setComponents] = useState([]);
  const [loading, setLoading] = useState(true);

  const fetchComponents = async () => {
    try {
      const res = await monitoring_api.get("/components"); 
      setComponents(res.data);
    } catch (error) {
      console.error("Failed to fetch components:", error);
      setComponents([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchComponents();
  }, [modelsUpdated ]);

  return (
    <Card className="flat-card dashboard-component">
      <Card.Body>
        <div className="border-bottom d-flex justify-content-between align-items-center mb-3" > 
          <h3>Monitoring Components</h3>
        </div>
            {loading ? (
              <div className="text-center">
                <Spinner animation="border" />
              </div>
            ) : (
              <Table className="dashboard-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Type</th>
                    <th>Status</th>
                  </tr>
                </thead>
                <tbody>
                  {components.length === 0 ? (
                    <tr>
                      <td colSpan="4" className="text-center">
                        No components found
                      </td>
                    </tr>
                  ) : (
                    components.map((comp) => (
                      <tr key={comp.instance_id}>
                        <td>{comp.name} ({comp.instance_id})</td>
                        <td>{comp.type || "N/A"}</td>
                        <td>
                          <span
                            className={`badge ${
                              comp.status === "active"
                                ? "bg-success"
                                : comp.status === "error"
                                ? "bg-danger"
                                : "bg-secondary"
                            }`}
                          >
                            {comp.status}
                          </span>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </Table>
            )}
          </Card.Body>
        </Card>
  );
}