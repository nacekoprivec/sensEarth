import React, { useState, useEffect, useMemo } from "react";
import { Card, Table, Spinner, Form, InputGroup, Button } from "react-bootstrap";
import FeatherIcon from "feather-icons-react";
import monitoring_api from "../../monitoring_api";

const TYPE_GROUPS = [
  { id: "all", label: "All" },
  { id: "scraper", label: "Scrapers" },
  { id: "model", label: "Models" },
  { id: "middleware", label: "Middleware" },
  { id: "database", label: "Database" },
  { id: "minio", label: "Minio" },
  { id: "other", label: "Other" },
];

const typeBadgeClass = (type) => {
  switch (type) {
    case "scraper":
      return "monitoring-type-badge monitoring-type-badge--scraper";
    case "model":
      return "monitoring-type-badge monitoring-type-badge--model";
    case "middleware":
      return "monitoring-type-badge monitoring-type-badge--middleware";
    case "database":
      return "monitoring-type-badge monitoring-type-badge--database";
    case "minio":
      return "monitoring-type-badge monitoring-type-badge--minio";
    default:
      return "monitoring-type-badge monitoring-type-badge--other";
  }
};

const statusBadgeClass = (status) => {
  switch (status) {
    case "active":
      return "bg-success";
    case "error":
      return "bg-danger";
    default:
      return "bg-secondary";
  }
};

export default function MonitoringDashboard({ modelsUpdated }) {
  const [components, setComponents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeType, setActiveType] = useState("all");

  const fetchComponents = async () => {
    try {
      const res = await monitoring_api.get("/components");
      const componentsData = Array.isArray(res.data) ? res.data : [];
      if (!Array.isArray(res.data)) {
        console.warn(
          "Monitoring components response was not an array, defaulting to []:",
          res.data
        );
      }
      setComponents(componentsData);
    } catch (error) {
      console.error("Failed to fetch components:", error);
      setComponents([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    setLoading(true);
    fetchComponents();
  }, [modelsUpdated]);

  const typeCounts = useMemo(() => {
    const counts = { all: components.length };
    TYPE_GROUPS.forEach((group) => {
      if (group.id !== "all") {
        counts[group.id] = components.filter(
          (comp) => (comp.type || "other") === group.id
        ).length;
      }
    });
    return counts;
  }, [components]);

  const filteredComponents = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();

    return components.filter((comp) => {
      const compType = comp.type || "other";
      const matchesType = activeType === "all" || compType === activeType;

      if (!matchesType) return false;
      if (!query) return true;

      const haystack = `${comp.name || ""} ${comp.instance_id || ""}`.toLowerCase();
      return haystack.includes(query);
    });
  }, [components, searchQuery, activeType]);

  return (
    <Card className="flat-card dashboard-component monitoring-dashboard">
      <Card.Body>
        <div className="monitoring-dashboard__header">
          <div className="monitoring-dashboard__title-row">
            <div>
              <h5 className="monitoring-dashboard__title mb-0">Monitoring Components</h5>
              <p className="monitoring-dashboard__subtitle mb-0">
                {loading
                  ? "Loading components..."
                  : `${filteredComponents.length} of ${components.length} shown`}
              </p>
            </div>

            <InputGroup size="sm" className="monitoring-dashboard__search">
              <InputGroup.Text>
                <FeatherIcon icon="search" size={14} />
              </InputGroup.Text>
              <Form.Control
                type="search"
                placeholder="Search components..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                aria-label="Search components"
              />
              {searchQuery && (
                <Button
                  variant="outline-secondary"
                  size="sm"
                  onClick={() => setSearchQuery("")}
                  aria-label="Clear search"
                >
                  <FeatherIcon icon="x" size={14} />
                </Button>
              )}
            </InputGroup>
          </div>

          <div className="monitoring-dashboard__filters" role="group" aria-label="Filter by component type">
            {TYPE_GROUPS.map((group) => (
              <button
                key={group.id}
                type="button"
                className={`monitoring-filter-btn${
                  activeType === group.id ? " monitoring-filter-btn--active" : ""
                }`}
                onClick={() => setActiveType(group.id)}
              >
                <span>{group.label}</span>
                <span className="monitoring-filter-btn__count">{typeCounts[group.id] ?? 0}</span>
              </button>
            ))}
          </div>
        </div>

        <div className="monitoring-dashboard__table-wrap">
          {loading ? (
            <div className="monitoring-dashboard__loading">
              <Spinner animation="border" />
            </div>
          ) : (
            <Table className="dashboard-table monitoring-table mb-0" hover>
              <colgroup>
                <col className="monitoring-table__col-name" />
                <col className="monitoring-table__col-instance" />
                <col className="monitoring-table__col-type" />
                <col className="monitoring-table__col-status" />
              </colgroup>
              <thead>
                <tr>
                  <th>Component</th>
                  <th>Instance</th>
                  <th>Type</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredComponents.length === 0 ? (
                  <tr>
                    <td colSpan="4" className="text-center monitoring-table__empty">
                      {components.length === 0
                        ? "No components found"
                        : "No components match your search or filter"}
                    </td>
                  </tr>
                ) : (
                  filteredComponents.map((comp) => (
                    <tr key={`${comp.name}-${comp.instance_id}`}>
                      <td className="monitoring-table__name">{comp.name}</td>
                      <td className="monitoring-table__instance">{comp.instance_id}</td>
                      <td>
                        <span className={typeBadgeClass(comp.type)}>
                          {comp.type || "other"}
                        </span>
                      </td>
                      <td>
                        <span
                          className={`badge monitoring-status-badge ${statusBadgeClass(
                            comp.status
                          )}`}
                        >
                          <span className="monitoring-status-badge__dot" />
                          {comp.status}
                        </span>
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
