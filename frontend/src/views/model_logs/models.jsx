import React, { useState, useEffect } from "react";
import { Card, Spinner } from "react-bootstrap";
import Button from "@mui/material/Button";
import AddIcon from "@mui/icons-material/Add";
import Dialog from "@mui/material/Dialog";
import DialogTitle from "@mui/material/DialogTitle";
import DialogContent from "@mui/material/DialogContent";
import DialogActions from "@mui/material/DialogActions";
import TextField from "@mui/material/TextField";
import DialogContentText from "@mui/material/DialogContentText";
import IconButton from "@mui/material/IconButton";
import DeleteIcon from "@mui/icons-material/Delete";
import PlayArrowIcon from "@mui/icons-material/PlayArrow";
import api from "../../api";
import monitoring_api from "../../monitoring_api";

export default function ModelsDashboard({ setModelsUpdated, selectedModel = null, onSelectModel = () => {} }) {
  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [openDialog, setOpenDialog] = useState(false);
  const [modelName, setModelName] = useState("");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("anomaly_detection_model");
  const [sensorIds, setSensorIds] = useState("1,2,3");
  const [jsonConfig, setJsonConfig] = useState(`{
  "anomaly_detection_alg": ["BorderCheck()"],
  "anomaly_detection_conf": [
    {
      "input_vector_size": 1,
      "warning_stages": [90, 5],
      "UL": 100,
      "LL": 0,
      "output": ["TerminalOutput()"],
      "output_conf": [{}]
    }
  ]
}`);

  const fetchModels = async () => {
    try {
      const res = await api.get("/models");
      setModels(Array.isArray(res.data) ? res.data : []);
    } catch (error) {
      console.error("Failed to fetch models:", error);
      setModels([]);
    }
    setLoading(false);
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleCreateModel = async () => {
    let parsedJson = {};
    try {
      parsedJson = JSON.parse(jsonConfig);
    } catch (err) {
      alert("Invalid JSON config");
      return;
    }

    const payload = {
      model_name: modelName,
      model_description: description || null,
      model_type: modelType || "anomaly_detection_model",
      sensor_id_list: sensorIds.split(",").map((id) => parseInt(id.trim())),
      model_parameters: parsedJson,
    };

    try {
      await api.post("/registerModel", payload);
      setOpenDialog(false);
      setModelsUpdated((v) => v + 1);
      setModelName("");
      setDescription("");
      setJsonConfig("{}");
      fetchModels();
    } catch (error) {
      console.error("Failed to create model:", error);
      alert("Failed to register model");
    }
  };

  return (
    <>
      <Card className="flat-card dashboard-component model-list-panel">
        <Card.Body className="d-flex flex-column model-list-panel__body">
          <div className="model-panel__header">
            <div>
              <h5 className="model-panel__title mb-0">Models</h5>
              <p className="model-panel__subtitle mb-0">Select a model to view details</p>
            </div>
            <div className="model-panel__header-actions">
              <IconButton className="btn-icon-small" color="primary" onClick={() => setOpenDialog(true)}>
                <AddIcon />
              </IconButton>
              <Button
                startIcon={<DeleteIcon />}
                color="error"
                className="btn-icon-small"
                onClick={async () => {
                  try {
                    if (confirm("Are you sure you want to delete all models?")) {
                      await api.delete("/models");
                      await Promise.all(
                        models.map((model) =>
                          monitoring_api.delete(
                            `/component?name=${encodeURIComponent(model.model_type)}&instance_id=${encodeURIComponent(model.name)}`
                          )
                        )
                      );
                      onSelectModel(null);
                      setModelsUpdated((v) => v + 1);
                      fetchModels();
                    }
                  } catch (error) {
                    console.error(error);
                  }
                }}
              />
            </div>
          </div>

          <div className="model-list flex-grow-1">
            {loading ? (
              <div className="text-center py-4">
                <Spinner animation="border" />
              </div>
            ) : models.length === 0 ? (
              <div className="text-muted text-center py-4 small">No models found</div>
            ) : (
              models.map((model) => (
                <div
                  key={model.model_id ?? model.name}
                  className={`model-list__item${
                    selectedModel?.model_id === model.model_id ? " model-list__item--selected" : ""
                  }`}
                  onClick={() => onSelectModel(model)}
                >
                  <div className="model-list__info">
                    <div className="model-list__name">{model.name}</div>
                    <div className="model-list__type">{model.model_type || "N/A"}</div>
                  </div>
                  <div className="model-list__actions">
                    <IconButton
                      size="small"
                      color="primary"
                      onClick={async (e) => {
                        e.stopPropagation();
                        try {
                          if (confirm("Run this model?")) {
                            await api.post("/runModel", { model_name: model.name });
                            setModelsUpdated((v) => v + 1);
                            fetchModels();
                          }
                        } catch (error) {
                          console.error(error);
                        }
                      }}
                    >
                      <PlayArrowIcon fontSize="small" />
                    </IconButton>
                    <IconButton
                      size="small"
                      color="error"
                      onClick={async (e) => {
                        e.stopPropagation();
                        try {
                          if (confirm("Delete this model?")) {
                            await api.delete(`/models/${encodeURIComponent(model.name)}`);
                            await monitoring_api.delete(
                              `/component?name=${encodeURIComponent(model.model_type)}&instance_id=${encodeURIComponent(model.name)}`
                            );
                            if (selectedModel?.model_id === model.model_id) onSelectModel(null);
                            setModelsUpdated((v) => v + 1);
                            fetchModels();
                          }
                        } catch (error) {
                          console.error(error);
                        }
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </div>
                </div>
              ))
            )}
          </div>
        </Card.Body>
      </Card>

      <Dialog open={openDialog} onClose={() => setOpenDialog(false)} maxWidth="sm" fullWidth>
        <DialogTitle>Register Model</DialogTitle>
        <DialogContent>
          <DialogContentText sx={{ mb: 2 }}>
            Register a new ML model and configure its sensors and parameters.
          </DialogContentText>
          <TextField
            label="Model Name"
            fullWidth
            margin="normal"
            value={modelName}
            onChange={(e) => setModelName(e.target.value)}
          />
          <TextField
            label="Description (optional)"
            fullWidth
            margin="normal"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
          <TextField
            label="Model Type"
            fullWidth
            margin="normal"
            value={modelType}
            onChange={(e) => setModelType(e.target.value)}
          />
          <TextField
            label="Sensor ID List (comma separated)"
            fullWidth
            margin="normal"
            value={sensorIds}
            onChange={(e) => setSensorIds(e.target.value)}
          />
          <TextField
            label="JSON Config Parameters"
            fullWidth
            margin="normal"
            multiline
            minRows={5}
            value={jsonConfig}
            onChange={(e) => setJsonConfig(e.target.value)}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenDialog(false)}>Cancel</Button>
          <Button variant="contained" onClick={handleCreateModel}>
            Register
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}
