import React, { useState, useEffect, useRef } from 'react';
import maplibregl from 'maplibre-gl';
import { Card, Typography, Box, CircularProgress } from '@mui/material';
import api from '../../api';

const center = [14.0, 46.0]; // [lng, lat]

export default function MapDashboard() {
  const mapContainer = useRef(null);
  const mapRef = useRef(null);

  const [sensors, setSensors] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchSensors();
  }, []);

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
      style: {version: 8, 
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
                type: s.name
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

      // Markers
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
      map.on('click', 'sensor-points', (e) => {
        const feature = e.features[0];
        const coords = feature.geometry.coordinates.slice();

        const { label, status, type, id } = feature.properties;

        new maplibregl.Popup()
          .setLngLat(coords)
          .setHTML(`
            <div>
              <strong>${label}</strong><br/>
              Type: ${type}<br/>
              Status: ${status}<br/>
              ID: ${id}
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

    return () => map.remove();
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
        style={{ width: '100%', height: '400px', borderRadius: '8px' }}
      />
    </Card>
  );
}