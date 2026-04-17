import React, { useState, useEffect } from "react";
import { alignPropType } from "react-bootstrap/esm/types";

export default function ChartSettingsDialog({ allSensors, selectedSensors, setSelectedSensors, days, setDays, resetChart }) {
  const [dialogOpen, setDialogOpen] = useState(false);
  const [tempDays, setTempDays] = useState(days);
  const [tempSelected, setTempSelected] = useState(selectedSensors);

  useEffect(() => {
    if (dialogOpen) {
      setTempDays(days);
      setTempSelected(selectedSensors);
    }
  }, [dialogOpen, days, selectedSensors]);

const toggleSensor = id => {
    const clickedSensor = allSensors.find(s => s.sensor_id === id);
    const selectedSensorsDetails = tempSelected.map(sid => allSensors.find(s => s.sensor_id === sid));

    if (selectedSensorsDetails.length > 0) {
      const existingType = selectedSensorsDetails[0].name;
      if (clickedSensor.name !== existingType && !tempSelected.includes(id)) {
        alert(`Sensors must be of the same type. Current type: "${existingType}", selected: "${clickedSensor.name}"`);
        return;
      }
    }

    setTempSelected(prev =>
      prev.includes(id) ? prev.filter(x => x !== id) : [...prev, id]
    );
  };
  
  const applySettings = () => {
    setSelectedSensors(tempSelected);
    setDays(tempDays);
    setDialogOpen(false);
  };

  return (
    <>
    <div className="border-bottom d-flex justify-content-between align-items-center mb-3" >
      <h3 >Sensor data overview</h3>
      <div style={{ marginLeft: "100px", marginBottom: "8px", gap: "8px", display: "flex" }}>
       <button className="btn-open" onClick={resetChart}>
        Reset
        </button>
          <button className="btn-open" onClick={() => setDialogOpen(true)}>
        Settings
      </button>
        </div>
      </div>
      {dialogOpen && (
        <div className="modal-overlay" onClick={() => setDialogOpen(false)}>
          <div className="settings-dialog" onClick={(e) => e.stopPropagation()}>
            <h3 className="dialog-title">Chart Settings</h3>
            <div className="settings-section">
              <label className="settings-label">Lookback Period</label>
              <input
                type="number"
                className="number-input"
                min={1}
                max={365}
                value={tempDays}
                onChange={e => setTempDays(Number(e.target.value))}
              />
            </div>

            <div className="settings-section">
              <label className="settings-label">Active Sensors</label>
              <div className="scroll-area">
                {allSensors.map(s => (
                  <label key={s.sensor_id} className="sensor-item">
                    <input
                      type="checkbox"
                      checked={tempSelected.includes(s.sensor_id)}
                      onChange={() => toggleSensor(s.sensor_id)}
                      style={{ marginRight: '12px' }}
                    />
                    <div className="sensor-info">
                      <strong>{s.sensor_label}</strong>
                      <span> {s.name} </span>
                       <span className="sensor-id">({s.sensor_id})</span>
                    </div>
                  </label>
                ))}
              </div>
            </div>

            <div className="btn-group">
            <button className="btn-open" style={{marginRight: 'auto'}} onClick={() => { setSelectedSensors([]); setDays(7);}} >
              Reset
             </button>
              <button className="btn-cancel" onClick={() => setDialogOpen(false)}>
                Cancel
              </button>
              <button className="btn-apply" onClick={applySettings}>
                Apply
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}