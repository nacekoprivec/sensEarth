import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { Card, Typography, Box, CircularProgress } from '@mui/material';
import IconButton from '@mui/material/IconButton';
import CloseIcon from '@mui/icons-material/Close';
import api from '../../api';
import monitoring_api from "../../monitoring_api";
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

  const [locationSensors, setLocationSensors] = useState([]);
  const [activeSensor, setActiveSensor] = useState(null);
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

  const loadSensorData = async (sensor) => {
    setActiveSensor(sensor);

    setLoadingMeasurements(true);

    const data = await fetchMeasurements([sensor.id], 30);

    setMeasurements(data);

    setLoadingMeasurements(false);

    fetchMonitoringData(sensor.id);
  };

  const getGroupStatus = (sensorGroup) => {
    if (sensorGroup.some((s) => s.status === 'error')) return 'error';
    if (sensorGroup.some((s) => s.status !== 'active')) return 'inactive';
    return 'active';
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
      const grouped = {};

      sensors
        .filter((s) => s.location)
        .forEach((s) => {
          const coords = JSON.parse(s.location).coordinates;
          const key = coords.join(",");

          if (!grouped[key]) {
            grouped[key] = {
              coordinates: coords,
              sensors: [],
            };
          }

          grouped[key].sensors.push({
            id: s.sensor_id,
            label: s.sensor_label,
            status: s.sensor_status,
            type: s.name,
          });
        });

      const geojson = {
        type: 'FeatureCollection',
        features: Object.values(grouped).map((group) => ({
          type: 'Feature',
          properties: {
            sensors: JSON.stringify(group.sensors),
            status: getGroupStatus(group.sensors),
            sensorCount: group.sensors.length,
          },
          geometry: {
            type: 'Point',
            coordinates: group.coordinates,
          },
        })),
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

          const sensorsAtLocation = JSON.parse(feature.properties.sensors);
          setLocationSensors(sensorsAtLocation);

          if (sensorsAtLocation.length === 1) {
            await loadSensorData(sensorsAtLocation[0]);
            return;
          }

          const popupNode = document.createElement('div');
          popupNode.style.minWidth = '180px';

          const title = document.createElement('div');
          title.innerHTML = `<strong>Select sensor type</strong>`;
          title.style.marginBottom = '8px';

          popupNode.style.minWidth = '220px';
          popupNode.style.maxWidth = '220px';

          popupNode.style.maxHeight = '240px';
          popupNode.style.overflowY = 'auto';

          popupNode.style.paddingRight = '4px';

          popupNode.appendChild(title);

          sensorsAtLocation.forEach((sensor) => {
            const button = document.createElement('button');
            button.innerText = `${sensor.type}`;
            button.style.display = 'block';
            button.style.width = '100%';
            button.style.marginBottom = '6px';
            button.style.padding = '6px';
            button.style.cursor = 'pointer';
            button.style.border = '1px solid #ccc';
            button.style.borderRadius = '4px';
            button.style.background =
              activeSensor?.id === sensor.id
                ? '#1976d2'
                : '#fff';

            button.style.color =
              activeSensor?.id === sensor.id
                ? '#fff'
                : '#000';

            button.onclick = async (event) => {
              event.stopPropagation();

              // reset all buttons
              popupNode.querySelectorAll('button').forEach((btn) => {
                btn.style.background = '#fff';
                btn.style.color = '#000';
              });

              // activate clicked button
              button.style.background = '#1976d2';
              button.style.color = '#fff';

              setLocationSensors(sensorsAtLocation);

              await loadSensorData(sensor);
            };

            popupNode.appendChild(button);
          });

          popupRef.current
            .setLngLat(coords)
            .setDOMContent(popupNode)
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
  <div style={{ position: "relative" }}>
    {/* MAP */}
    <div
      ref={mapContainer}
      style={{
        width: '100vw',
        height: '350px',
        backgroundSize: 'cover',
        borderRadius: '8px',

        position: 'relative',
        left: '50%',
        transform: 'translateX(-50%)',

        marginBottom: '20px',
        marginTop: '-15px'
      }}
    />

    {activeSensor && (
      <div
        style={{
          position: 'absolute',
          top: 20,
          left: 20,
          width: '340px',
          maxHeight: '380px',
          background: 'rgba(255,255,255,0.95)',
          borderRadius: '8px',
          boxShadow: '0 4px 12px rgba(0,0,0,0.2)',
          padding: '12px',
          zIndex: 10,
          overflow: 'hidden',
        }}
      >
        <Typography variant="subtitle1" gutterBottom sx={{ pr: 4 }}>
          Measurements — {activeSensor.label}
        </Typography>
        <IconButton
          size="small"
          onClick={() => {
            setActiveSensor(null);
            setLocationSensors([]);
            setMeasurements([]);
            setSensorMetrics([]);
          }}
          style={{ position: 'absolute', top: 6, right: 6 }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>

        <Box mb={1}>
          <Typography variant="body2" color="textSecondary">
            Status: {activeSensor.status}
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Type: {activeSensor.type}
          </Typography>
        </Box>

        {loadingMeasurements ? (
          <CircularProgress size={20} />
        ) : (
          <div
            style={{
              maxHeight: '120px',
              overflowY: 'auto',
              border: '1px solid #eee',
              borderRadius: '6px',
            }}
          >
            <table style={{ width: '100%', fontSize: '0.75rem' }}>
              <thead style={{ position: 'sticky', top: 0, background: '#fafafa' }}>
                <tr>
                  <th style={{ textAlign: 'left', padding: '6px' }}>Timestamp</th>
                  <th style={{ textAlign: 'right', padding: '6px' }}>Value</th>
                </tr>
              </thead>
              <tbody>
                {measurements.map((m, i) => (
                  <tr key={i}>
                    <td style={{ padding: '6px' }}>
                      {new Date(m.timestamp_utc).toLocaleString()}
                    </td>
                    <td style={{ padding: '6px', textAlign: 'right' }}>
                      {m.value}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <Box mt={1}>
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
                maxHeight: '120px',
                overflowY: 'auto',
                border: '1px solid #eee',
                borderRadius: '6px',
              }}
            >
              <table style={{ width: '100%', fontSize: '0.75rem' }}>
                <thead style={{ position: 'sticky', top: 0, background: '#fafafa' }}>
                  <tr>
                    <th style={{ textAlign: 'left', padding: '6px' }}>Metric</th>
                    <th style={{ textAlign: 'right', padding: '6px' }}>Value</th>
                  </tr>
                </thead>
                <tbody>
                  {sensorMetrics.map((metric, i) => (
                    <tr key={i}>
                      <td style={{ padding: '6px' }}>{metric.metric_name}</td>
                      <td style={{ padding: '6px', textAlign: 'right' }}>
                        {metric.value}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Box>
      </div>
    )}
  </div>
);}