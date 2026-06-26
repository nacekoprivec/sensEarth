import React, { useEffect, useState } from "react";
import api from "../../api";
import ModelChartSettings from "./model_chart/ModelChartSettings";
import ModelsDashboard from "./models";
import ModelLogs from "./model_logs";

export default function Models() {
  const [allSensors, setAllSensors] = useState([]);
  const [modelsUpdated, setModelsUpdated] = useState(0);
  const [selectedModel, setSelectedModel] = useState(null);

  useEffect(() => {
    api
      .get("/sensors")
      .then((res) => setAllSensors(Array.isArray(res.data) ? res.data : []))
      .catch(() => setAllSensors([]));
  }, []);

  return (
    <div className="model-logs-page">
      <ModelChartSettings allSensors={allSensors} refreshKey={modelsUpdated} />
      <div className="model-logs-layout">
        <ModelsDashboard
          setModelsUpdated={setModelsUpdated}
          selectedModel={selectedModel}
          onSelectModel={setSelectedModel}
        />
        <ModelLogs refreshKey={modelsUpdated} selectedModel={selectedModel} />
      </div>
    </div>
  );
}
