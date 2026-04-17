import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Table, Spinner } from "react-bootstrap";
import Button from '@mui/material/Button';
import AddIcon from '@mui/icons-material/Add';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import DialogContentText from '@mui/material/DialogContentText';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import PlayArrowIcon from '@mui/icons-material/PlayArrow';
import api from '../../api';
import monitoring_api from "../../monitoring_api";

export default function ModelsDashboard({ setModelsUpdated }) {

  const [models, setModels] = useState([]);
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState([]);

  const [openDialog, setOpenDialog] = useState(false);

  const [modelName, setModelName] = useState("");
  const [description, setDescription] = useState("");
  const [modelType, setModelType] = useState("anomaly_detection_model");
  const [sensorIds, setSensorIds] = useState("1,2,3");
  const [results, setResults] = useState()
  const [modelResults, setModelResults] = useState(null);
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

  const transformResults = (results) => {
    return Object.entries(results).map(([sensorId, entries]) => {
      return {
        sensorId,
        rows: entries.map(([measurement, result]) => ({
          timestamp: measurement[0],
          value: measurement[1],
          message: result[0],
          code: result[1],
          suggestedValue: result.length > 2 ? result[2] : undefined
        }))
      };
    });
  };

  const fetchModels = async () => {
    try {
      const res = await api.get("/models");
      setModels(res.data);
      console.log("Fetched models:", res.data);
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
      sensor_id_list: sensorIds.split(",").map(id => parseInt(id.trim())),
      model_parameters: parsedJson
    };

    try {
      console.log("Registering model with payload:", payload);
      await api.post("/registerModel", payload);
      setOpenDialog(false);

      setModelsUpdated(v => v + 1);

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
      <Card className="flat-card dashboard-component">
        <Card.Body>
          <Card>
            <Card.Body>
              <div className="border-bottom d-flex justify-content-between align-items-center mb-3" >
                <h3>Models</h3>
                <div className="d-flex align-items-center">
                  <IconButton
                    className='btn-icon-small'
                    color="primary"
                    onClick={() => setOpenDialog(true)}
                  >
                    <AddIcon />
                  </IconButton>

                  <Button
                    startIcon={<DeleteIcon />}
                    color="error"
                    className='btn-icon-small'
                    onClick={async () => {
                      try {
                        if (confirm('Are you sure you want to delete all models?')) {
                          await api.delete('/models');
                          // Delete all model components in monitoring system
                          await Promise.all(
                            models.map(model =>
                              monitoring_api.delete(
                                `/component?name=${encodeURIComponent(model.model_type)}&instance_id=${encodeURIComponent(model.name)}`
                              )
                            )
                          );
                          setModelsUpdated(v => v + 1);
                          fetchModels();
                        }
                      } catch (error) {
                        console.error(error);
                      }
                    }}
                  >
                  </Button>
                </div>
              </div>

              {loading ? (
                <div className="text-center">
                  <Spinner animation="border" />
                </div>
              ) : (
                <Table hover responsive size="sm" className="align-middle mb-0">
                  <thead className="table-light">
                    <tr>
                      <th style={{ width: "40%" }}>Model</th>
                      <th style={{ width: "35%" }}>Type</th>
                      <th style={{ width: "25%" }} className="text-end">Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {models.length === 0 ? (
                      <tr>
                        <td colSpan="3" className="text-center text-muted py-3">
                          No models found
                        </td>
                      </tr>
                    ) : (
                      models.map((model) => (
                        <tr key={model.models_id} className="model-row">

                          {/* Name */}
                          <td>
                            <div className="fw-semibold text-truncate">
                              {model.name}
                            </div>
                          </td>

                          {/* Type */}
                          <td>
                            <span className="badge bg-light text-dark border">
                              {model.model_type || "N/A"}
                            </span>
                          </td>

                          {/* Actions */}
                          <td>
                            <div className="d-flex justify-content-end gap-1">

                              <IconButton
                                size="small"
                                color="primary"
                                onClick={async () => {
                                  try {
                                    if (confirm('Run this model?')) {
                                      const res = await api.post(`/runModel`, {
                                        model_name: model.name,
                                      });

                                      setModelResults(res.data);
                                      setLogs(prev => [
                                        ...prev,
                                        `> Running ${model.name}...`,
                                        `✔ Model finished`
                                      ]);

                                      setModelsUpdated(v => v + 1);
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
                                onClick={async () => {
                                  try {
                                    if (confirm('Delete this model?')) {
                                      await api.delete(`/models/${encodeURIComponent(model.name)}`);
                                      await monitoring_api.delete(
                                        `/component?name=${encodeURIComponent(model.model_type)}&instance_id=${encodeURIComponent(model.name)}`
                                      );
                                      setModelsUpdated(v => v + 1);
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
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>

          <Card className="flat-card scroll-area">
            <Card.Body>
              <h5>Model results</h5>

              {modelResults && (
                <Card className="mt-3">
                  <Card.Body>
                    <div className="model-results-container">
                      {transformResults(modelResults.results).map(sensor => {
                        const errors = sensor.rows.filter(r => r.code === -1).length;
                        return (
                          <div key={sensor.sensorId} className="sensor-block">

                            {/* Header */}
                            <div className="dashboard-header">
                              Sensor {sensor.sensorId} | {errors}/{sensor.rows.length} errors
                            </div>

                            {/* Table */}
                            <div className="dashboard-table">
                              {sensor.rows.slice(0, 15).map((row, i) => (
                                <div key={i} className="dashboard-row">
                                  <span className="dashboard-cell">
                                    {row.timestamp.toFixed(2)}
                                  </span>

                                  <span className="dashboard-cell">
                                    {row.value}
                                  </span>

                                  <span className={`dashboard-cell ${row.suggestedValue !== undefined ? 'dashboard-suggested' : row.message === 'OK' ? 'dashboard-success' : row.message.includes('Warning') ? 'dashboard-warning' : 'dashboard-error'}`}>
                                    {row.suggestedValue !== undefined ? `Suggested: ${row.suggestedValue}` : row.message}
                                  </span>
                                </div>
                              ))}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </Card.Body>
                </Card>
              )}
            </Card.Body>
          </Card>

        </Card.Body>
      </Card>

      {/* Add Model Dialog */}
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
            placeholder={`{
            "anomaly_detection_alg": ["BorderCheck()"],
            "anomaly_detection_conf": [
              {
                "input_vector_size": 1,
                "warning_stages": [2.5, 0.0],
                "UL": 3.0,
                "LL": -0.4,
                "output": ["TerminalOutput()"],
                "output_conf": [{}]
              }
            ]
          }`}
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