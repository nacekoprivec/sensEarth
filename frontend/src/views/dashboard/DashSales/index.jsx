import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Spinner } from "react-bootstrap";
import api from '../../../api';
import MonitoringDashboard from './monitoring';
import ModelsDashboard from './models';
import EventsDashboard from './events';
import SensorChart from "./chart/SensorChart";
import ChartSettingsModal from "./chart/ChartSettingsModal";
import ModelChartSettings from "./model_chart/ModelChartSettings";
import LatestMeasurementsDashboard from './latest_measurements';
import IngestionStatus from './ingestion_status';
import DataOverview from './data_overiew';
import HealthOverview from './health_overview';
import ModelRuns from './models_overview';
import MapDashboard from './map';
//-----------------------|| DASHBOARD SENSEARTH ||-----------------------//

async function fetchMeasurements(sensorIDs = [], days = 0) {
  try {
    const params = new URLSearchParams();
    sensorIDs.forEach(id => params.append("sensorIDs", id));
    if (days) params.append("days", days);

    console.log("Fetching measurements with params:", { params: params.toString() });

    const res = await api.get(`/measurements?${params.toString()}`);

    const measurements = res.data;
    console.log("Fetched measurements:", measurements);

    return measurements;
  } catch (error) {
    console.error("Failed to fetch measurements:", error);
    return [];
  }
}

export default function DashboardSales() {
  const [sensors, setSensors] = useState([]);
  const [loading, setLoading] = useState(true);
  const [allSensors, setAllSensors] = useState([]); // [{id, label}]
  const [selectedSensors, setSelectedSensors] = useState([]);
  const [days, setDays] = useState(30);
  const [measurements, setMeasurements] = useState([]);
  const [chartReset, setChartReset] = useState(0);

  // If specific component was updated, then refresh 
  const [modelsUpdated, setModelsUpdated] = useState(0);

  // Load all sensors once
  const fetchSensorsAll = async () => {
    try {
      const res = await api.get("/sensors");
      setAllSensors(res.data);
      console.log("Fetched all sensors:", res.data);
    } catch (error) {
      console.error("Failed to fetch sensors:", error);
      setAllSensors([]);
    }
    setLoading(false);
  };

  const resetChart = () => {
    setSelectedSensors([]);
    setDays(30);
    setMeasurements([]);
    setChartReset(v => v + 1);
  };

  useEffect(() => {
    fetchSensorsAll();
  }, []);

  // Fetch measurements whenever selectedSensors or days change
  useEffect(() => {
    if (selectedSensors.length === 0) return;
    fetchMeasurements(selectedSensors, days)
      .then(data => setMeasurements(data))
      .catch(err => console.error(err));
  }, [selectedSensors, days]);

  const fetchSensors = async () => {
    try {
      const res = await api.get("/measurements?limit=10");
      setSensors(res.data);
    } catch (error) {
      console.error("Failed to fetch sensors:", error);
      setSensors([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchSensors();
  }, []);

  return (
    <>
      <MapDashboard />

      <div className="system-overview">
        <IngestionStatus modelsUpdated={modelsUpdated} />
        <DataOverview refreshKey={modelsUpdated} />
        <HealthOverview refreshKey={modelsUpdated} />
        <ModelRuns refreshKey={modelsUpdated} />
      </div>
      <div className="dashboard-grid">

        <LatestMeasurementsDashboard sensors={sensors} loading={loading} />
        <Card className="flat-card" style={{ gridColumn: "span 1" }}>
          <Card.Body>
            <ChartSettingsModal
              allSensors={allSensors}
              selectedSensors={selectedSensors}
              setSelectedSensors={setSelectedSensors}
              days={days}
              setDays={setDays}
              onClose={() => setShowSettings(false)}
              resetChart={resetChart}
            />
            <SensorChart key={chartReset} measurements={measurements} />
          </Card.Body>
        </Card>

        <MonitoringDashboard modelsUpdated={modelsUpdated} />
        <ModelsDashboard setModelsUpdated={setModelsUpdated} />
        <Card className="flat-card" style={{ gridColumn: "span 1" }}>
          <Card.Body>
            <ModelChartSettings allSensors={allSensors} />
          </Card.Body>
        </Card>
        <EventsDashboard />
      </div>
    </>
  );
}