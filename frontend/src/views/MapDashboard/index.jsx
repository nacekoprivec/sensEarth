import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { Card, Typography, Box, CircularProgress } from '@mui/material';
import api from '../../api';
import monitoring_api from '../../monitoring_api';
import 'maplibre-gl/dist/maplibre-gl.css';

const center = [14.0, 46.0]; // [lng, lat]

async function fetchMeasurements(sensorIDs = [], days = 0) {
  try {
    const params = new URLSearchParams();
    sensorIDs.forEach(id => params.append("sensorIDs", id));
    if (days) params.append("days", days);

    console.log("Fetching measurements with params for map:", { params: params.toString() });

    const res = await api.get(`/measurements?${params.toString()}`);

    const measurements = res.data;
    console.log("Fetched measurements:", measurements);

    return measurements; // Expecting an array of { timestamp_utc, value, sensor_id, sensor_label, location }

  } catch (error) {
    console.error("Failed to fetch measurements:", error);
    return [];
  }
}



export default function MapDashboard() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);
  const popupRef = useRef(null);

  const [sensors, setSensors] = useState([]);
  const [loading, setLoading] = useState(true);

  const [selectedSensor, setSelectedSensor] = useState(null);
  const [measurements, setMeasurements] = useState([]);
  const [loadingMeasurements, setLoadingMeasurements] = useState(false);
  const [sensorMetrics, setSensorMetrics] = useState([]);
  const [loadingMetrics, setLoadingMetrics] = useState(false);

  useEffect(() => {
    fetchSensors();
  }, []);

  const fetchMonitoringData = async (sensorID) => {
    setLoadingMetrics(true);
    try {
      const metricsRes = await monitoring_api.get("/metrics");
      const metrics = Array.isArray(metricsRes.data) ? metricsRes.data : [];
      const sensorIdStr = String(sensorID);

      const matchedMetrics = metrics.filter(
        (m) =>
          typeof m.metric_name === "string" &&
          m.metric_name.includes(`sensor_id=${sensorIdStr}`)
      );

      setSensorMetrics(matchedMetrics);
    } catch (error) {
      console.error("Failed to fetch monitoring metrics:", error);
      setSensorMetrics([]);
    } finally {
      setLoadingMetrics(false);
    }
  };

  const fetchSensors = async () => {
    try {
      const response = await api.get('/sensors');
      setSensors(response.data);
    } catch (error) {
      console.error('Failed to fetch sensors:', error);
    } finally {
      setLoading(false);
    }
  };

  // Initialize map
  useEffect(() => {
    if (!mapContainer.current || loading) return;

    const map = new maplibregl.Map({
      container: mapContainer.current,
      style: {
        version: 8,
        sources: {
          osm: {
            type: 'raster',
            tiles: [
              'https://a.tile.openstreetmap.org/{z}/{x}/{y}.png'
            ],
            tileSize: 256
          }
        },
        layers: [
          {
            id: 'osm-tiles',
            type: 'raster',
            source: 'osm'
          }
        ]
      },
      center: center,
      zoom: 8
    });

    mapRef.current = map;

    map.addControl(new maplibregl.NavigationControl(), 'top-right');
    popupRef.current = new maplibregl.Popup();

    map.on('load', () => {
      const geojson = {
        type: 'FeatureCollection',
        features: sensors
          .filter(s => s.location)
          .map(s => {
            const coords = JSON.parse(s.location).coordinates;

            return {
              type: 'Feature',
              properties: {
                id: s.sensor_id,
                label: s.sensor_label,
                status: s.sensor_status,
                type: s.name,
              },
              geometry: {
                type: 'Point',
                coordinates: coords
              }
            };
          })
      };

      map.addSource('sensors', {
        type: 'geojson',
        data: geojson
      });

      // Sensor points
      map.addLayer({
        id: 'sensor-points',
        type: 'circle',
        source: 'sensors',
        paint: {
          'circle-radius': 6,
          'circle-color': [
            'match',
            ['get', 'status'],
            'active', '#28a745',
            'inactive', '#ffc107',
            'error', '#dc3545',
            '#6c757d'
          ],
          'circle-stroke-width': 1,
          'circle-stroke-color': '#fff'
        }
      });

      // Click popup 
          map.on('click', 'sensor-points', async (e) => {
        const feature = e.features[0];
        const coords = feature.geometry.coordinates.slice();

        const { label, status, type, id } = feature.properties;

        setSelectedSensor({ id, label });
        setSensorMetrics([]);

        setLoadingMeasurements(true);
        const data = await fetchMeasurements([id], 30);
        setMeasurements(data);
        setLoadingMeasurements(false);

        fetchMonitoringData(id);

        popupRef.current
          .setLngLat(coords)
          .setHTML(`
            <div style="font-size: 13px;">
              <strong>${label}</strong><br/>
              <div>Type: ${type}</div>
              <div>Status: ${status}</div>
            </div>
          `)
          .addTo(map);
      });

      // Cursor pointer
      map.on('mouseenter', 'sensor-points', () => {
        map.getCanvas().style.cursor = 'pointer';
      });

      map.on('mouseleave', 'sensor-points', () => {
        map.getCanvas().style.cursor = '';
      });
    });

    return () => {
      map.remove();
    };
  }, [loading, sensors]);

  if (loading) {
    return (
      <Card sx={{ p: 2 }}>
        <Box display="flex" justifyContent="center" alignItems="center" height={400}>
          <CircularProgress />
        </Box>
      </Card>
    );
  }

  return (
    <Card sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        Sensor Locations
      </Typography>

      <div
        ref={mapContainer}
        style={{ width: '100%', height: '500px', borderRadius: '8px' }}
      />
      {selectedSensor && (
      <Box mt={2}>
        <Typography variant="subtitle1" gutterBottom>
          Measurements — {selectedSensor.label}
        </Typography>

        {loadingMeasurements ? (
          <CircularProgress size={20} />
        ) : (
          <div
            style={{
              maxHeight: "250px",
              overflowY: "auto",
              border: "1px solid #eee",
              borderRadius: "6px",
            }}
          >
            <table style={{ width: "100%", fontSize: "0.8rem" }}>
              <thead style={{ position: "sticky", top: 0, background: "#fafafa" }}>
                <tr>
                  <th style={{ textAlign: "left", padding: "6px" }}>Timestamp</th>
                  <th style={{ textAlign: "right", padding: "6px" }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {measurements.map((m, i) => (
                  <tr key={i}>
                    <td style={{ padding: "6px" }}>
                      {new Date(m.timestamp_utc).toLocaleString()}
                    </td>
                    <td style={{ padding: "6px", textAlign: "right" }}>
                      {m.value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <Box mt={2}>
          <Typography variant="subtitle2" gutterBottom>
            Sensor metrics
          </Typography>

          {loadingMetrics ? (
            <CircularProgress size={20} />
          ) : sensorMetrics.length === 0 ? (
            <Typography variant="body2" color="textSecondary">
              No sensor metrics found.
            </Typography>
          ) : (
            <div
              style={{
                maxHeight: "250px",
                overflowY: "auto",
                border: "1px solid #eee",
                borderRadius: "6px",
                marginTop: 8,
              }}
            >
              <table style={{ width: "100%", fontSize: "0.8rem" }}>
                <thead style={{ position: "sticky", top: 0, background: "#fafafa" }}>
                  <tr>
                    <th style={{ textAlign: "left", padding: "6px" }}>Metric</th>
                    <th style={{ textAlign: "right", padding: "6px" }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {sensorMetrics.map((metric, i) => (
                    <tr key={i}>
                      <td style={{ padding: "6px" }}>{metric.metric_name}</td>
                      <td style={{ padding: "6px", textAlign: "right" }}>
                        {metric.value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Box>
      </Box>
    )}
    </Card>
  );
}