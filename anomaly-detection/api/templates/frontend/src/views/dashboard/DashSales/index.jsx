import React, { useState, useEffect } from 'react';
import { Row, Col, Card, Spinner, Accordion } from 'react-bootstrap';
import IconButton from '@mui/material/IconButton';
import DeleteIcon from '@mui/icons-material/Delete';
import Button from '@mui/material/Button';
import AddIcon from '@mui/icons-material/Add';
import Dialog from '@mui/material/Dialog';
import DialogTitle from '@mui/material/DialogTitle';
import DialogContent from '@mui/material/DialogContent';
import DialogActions from '@mui/material/DialogActions';
import TextField from '@mui/material/TextField';
import api from '../../../api';
import DialogContentText from '@mui/material/DialogContentText';

//-----------------------|| DASHBOARD SALES ||-----------------------//
export default function DashSales() {
  const [selectedMethod, setSelectedMethod] = useState('border_check.json');
  const [overrides, setOverrides] = useState({});
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [detectors, setDetectors] = useState([]);
  

  const fetchDetectors = async () => {
    try {
      const res = await api.get('/detectors');
      setDetectors(res.data);
    } catch {
      setDetectors([]);
    }
  }

  useEffect(() => {
    fetchDetectors();
  }, []);

  function getNestedValue(obj, path, fallback) {
    return path.split('.').reduce((acc, k) => (acc ? acc[k] : undefined), obj) ?? fallback;
  }

  function ConfigEditor({ data, overrides, setOverrides, parentKey = '' }) {
    const handleChange = (key, value) => {
      setOverrides((prev) => {
        const keys = parentKey ? parentKey.split('.') : [];
        let updated = { ...prev };
        let ref = updated;
        keys.forEach((k) => {
          ref[k] = { ...ref[k] };
          ref = ref[k];
        });
        ref[key] = value;
        return updated;
      });
    };

    return (
      <div style={{ paddingLeft: parentKey ? 15 : 0, borderLeft: parentKey ? '1px solid #eee' : 'none' }}>
        {Object.entries(data).map(([key, value]) => {
          const path = parentKey ? `${parentKey}.${key}` : key;

          if (typeof value === 'string' || typeof value === 'number') {
            return (
              <div className="mb-2" key={path}>
                <label>{key}</label>
                <input
                  type="text"
                  className="form-control"
                  value={getNestedValue(overrides, path, value)}
                  onChange={(e) => handleChange(key, e.target.value)}
                />
              </div>
            );
          }

          if (Array.isArray(value)) {
            if (value.every((v) => typeof v === 'object' && v !== null)) {
              return (
                <div key={path} className="mb-2">
                  <label>{key}</label>
                  {value.map((item, index) => (
                    <ConfigEditor
                      key={`${path}.${index}`}
                      data={item}
                      overrides={overrides?.[key]?.[index] ?? item}
                      setOverrides={(newOverrides) => {
                        setOverrides((prev) => ({
                          ...prev,
                          [key]: [...(prev?.[key] ?? []).slice(0, index), newOverrides, ...(prev?.[key] ?? []).slice(index + 1)]
                        }));
                      }}
                      parentKey={`${path}.${index}`}
                    />
                  ))}
                </div>
              );
            } else {
              return (
                <div className="mb-2" key={path}>
                  <label>{key}</label>
                  <input
                    type="text"
                    className="form-control"
                    value={overrides?.[key]?.join(',') ?? value.join(',')}
                    onChange={(e) => handleChange(key, e.target.value.split(','))}
                  />
                </div>
              );
            }
          }

          if (typeof value === 'object' && value !== null) {
            return (
              <div key={path} className="mb-2">
                <label>{key}</label>
                <ConfigEditor
                  data={value}
                  overrides={overrides?.[key] ?? value}
                  setOverrides={(newOverrides) => {
                    setOverrides((prev) => ({ ...prev, [key]: newOverrides }));
                  }}
                  parentKey={path}
                />
              </div>
            );
          }

          return null;
        })}
      </div>
    );
  }

  // Detector Card Component
  function DetectorCard({ detector, fetchDetectors }) {
    const [selectedMethod, setSelectedMethod] = useState(detector.config_name);
    const [config, setConfig] = useState(null);
    const [overrides, setOverrides] = useState({});
    const [response, setResponse] = useState('');
    const [loading, setLoading] = useState(false);
    const [timestamp, setTimestamp] = useState('');
    const [ftrVector, setFtrVector] = useState('');

    useEffect(() => {
      async function fetchConfig() {
        try {
          const res = await api.get(`/configuration/${selectedMethod}`);
          setConfig(res.data);
          setOverrides(res.data);
        } catch {
          setConfig(null);
        }
      }
      fetchConfig();
    }, [selectedMethod]);

    // const handleSaveConfig = async () => {
    //   try {
    //     const res = await api.post(`/configuration/${selectedMethod}`, overrides, detector.id);
    //     setResponse(res.data);
    //   } catch (error) {
    //     setResponse('Error: ' + error.message);
    //   }
    // };

    const handleRun = async () => {
      setLoading(true);
      try {
        const res = await api.post(
          `/detectors/${detector.id}/detect_anomaly/?timestamp=${encodeURIComponent(timestamp)}&ftr_vector=${encodeURIComponent(ftrVector)}`
        );
        setResponse(res.data);
      } catch (error) {
        setResponse('Error: ' + error.message);
      }
      setLoading(false);
    };

    return (
      <Card className="mb-3">
        <Card.Body>
          <Accordion flush>
            <Accordion.Item eventKey="0">
              <Accordion.Header>
                <div className="d-flex justify-content-between align-items-center w-100">
                  <span>
                    <strong>{detector.name}</strong> Detector
                  </span>
                  <span
                    className={`badge ${
                      detector.status === 'active'
                        ? 'bg-success'
                        : detector.status === 'error'
                        ? 'bg-danger'
                        : 'bg-secondary'
                    }`}
                  >
                    {detector.status}
                  </span>
                </div>
              </Accordion.Header>
              <Accordion.Body>
                <div className="mb-2">
                  <label>Timestamp</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder='e.g. "123.456"'
                    value={timestamp}
                    onChange={(e) => setTimestamp(e.target.value)}
                  />
                </div>
                <div className="mb-2">
                  <label>Feature Vector</label>
                  <input
                    type="text"
                    className="form-control"
                    placeholder="e.g. 1,2,3,4"
                    value={ftrVector}
                    onChange={(e) => setFtrVector(e.target.value)}
                  />
                </div>
                {config && <ConfigEditor data={config} overrides={overrides} setOverrides={setOverrides} />}

                <div className="mb-3 d-flex align-items-center">
                  {/* <button className="btn btn-success me-2" onClick={handleSaveConfig}>
                    Save Config
                  </button> */}

                  <button
                    className={`btn ${detector.status === 'inactive' ? 'btn-success' : 'btn-danger'} me-2`}
                    onClick={async () => {
                      try {
                        const newStatus = detector.status === 'inactive' ? 'active' : 'inactive';
                        await api.put(`/detectors/${detector.id}/${newStatus}`);
                        fetchDetectors();
                      } catch (error) {
                        console.error(error);
                      }
                    }}
                  >
                    {detector.status === 'inactive' ? 'Activate' : 'Deactivate'}
                  </button>

                  <Button
                    startIcon={<DeleteIcon />}
                    color="error"
                    onClick={async () => {
                      try {
                        if (confirm('Are you sure you want to delete this detector?')) {
                          await api.delete(`/detectors/${detector.id}`);
                          fetchDetectors();
                        }
                      } catch (error) {
                        console.error(error);
                      }
                    }}
                    className="me-2"
                  ></Button>

                  {loading ? (
                    <Spinner animation="border" style={{ marginLeft: 'auto' }} />
                  ) : (
                    <button className="btn btn-success ms-auto" onClick={handleRun}>
                      Run
                    </button>
                  )}
                </div>

                {response !== '' && (
                  <div className="mt-2">
                    <strong>API Response:</strong> {JSON.stringify(response)}
                  </div>
                )}
              </Accordion.Body>
            </Accordion.Item>
          </Accordion>
        </Card.Body>
      </Card>
    );
  }

function JsonPopupButton({ onSave, fetchDetectors }) {
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [selectedMethod, setSelectedMethod] = useState('');
  const [jsonText, setJsonText] = useState('');

  const handleOpen = () => {
    setOpen(true);
    setJsonText('');
  };

  const handleClose = () => {
    setOpen(false);
    setError('');
  };

  const handleSave = async () => {
    try {
      const parsed = JSON.parse(jsonText);
      parsed.config_name = selectedMethod; // include config in posted data
      parsed.name = name;
      parsed.description = description;
      
      console.log('Saving detector:', parsed);
      onSave(parsed);
      await api.post('/detectors/create', parsed);
      setOpen(false);
      fetchDetectors();
    } catch (e) {
      setError('Invalid JSON');
    }
  };

  const RetrieveConfig = async (configName) => {
    try {
      const res = await api.get(`/configuration/${configName}`);
      return res.data;
    } catch {
      return null;  
    }
  };

  const ConfigDropdown = () => {
    const [availableConfigs, setAvailableConfigs] = useState([]);

    useEffect(() => {
      async function fetchAvailableConfigs() {
        try {
          const res = await api.get('/available_configs');
          setAvailableConfigs(res.data);
        } catch {
          setAvailableConfigs([]);
        }
      }
      fetchAvailableConfigs();
    }, []);


    return (
      <select
  value={selectedMethod}
  onChange={async (e) => {
    const val = e.target.value;
    setSelectedMethod(val);

    const res = await RetrieveConfig(val);

    setJsonText((prev) => {
      try {
        const parsed = prev ? JSON.parse(prev) : {};
        return JSON.stringify(
          { ...parsed, anomaly_detection_alg: [val], config_data: res },
          null,
          2
        );
      } catch {
        return JSON.stringify(
          { anomaly_detection_alg: [val], config_data: res },
          null,
          2
        );
      }
    });
  }}
  className="form-control mb-3"
>
  <option value="">Select configuration...</option>
  {availableConfigs.map((ac) => (
    <option key={ac.name} value={ac.filename}>
      {ac.filename}
    </option>
  ))}
</select>
    );
  };

  return (
    <>
      <IconButton color="primary" onClick={handleOpen}>
        <AddIcon />
      </IconButton>

      <Dialog open={open} onClose={handleClose} fullWidth maxWidth="sm">
        <DialogTitle>Create New Detector</DialogTitle>
        <DialogContent>
          <TextField
            label="Name"
            value={name}
            onChange={(e) => setName(e.target.value)}
            fullWidth
            variant="outlined"
            size="small"
            className="mt-4 mb-4"
            required
            error={!!error}
            helperText={error ? 'Name is required' : ''}
          />

          <TextField
            label="Description"
            multiline
            rows={3}
            fullWidth
            variant="outlined"
            value={description}
            onChange={(e) => setDescription(e.target.value)}
            error={!!error}
            className="mb-4"
          />

          <DialogContentText>Select Configuration</DialogContentText>
          <ConfigDropdown />

          <TextField
            multiline
            rows={10}
            fullWidth
            variant="outlined"
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            required
            error={!!error}
            className="mb-4"
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
          <Button onClick={handleClose}>Cancel</Button>
          <Button variant="contained" color="primary" onClick={handleSave}>
            Save
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

  return (
    <Row>
      <Col md={12} xl={6} className="mb-3">
        <Card className="flat-card">
          <Card.Body>
            <div className="d-flex align-items-center mb-3">
              <h1 className="card-title me-3">Anomaly Detectors</h1>
              <JsonPopupButton onSave={(json) => console.log(json)} fetchDetectors={fetchDetectors} />
              <Button
                startIcon={<DeleteIcon />}
                color="error"
                onClick={async () => {
                  try {
                    if (confirm('Are you sure you want to delete all detectors?')) {
                      await api.delete('/detectors');
                      fetchDetectors();
                    }
                  } catch (error) {
                    console.error(error);
                  }
                }}
                className="ms-2"
              />
            </div>
            {detectors.map((det) => (
              <DetectorCard key={det.id} detector={det} fetchDetectors={fetchDetectors} />
            ))}
          </Card.Body>
        </Card>
      </Col>
    </Row>
  );
}
