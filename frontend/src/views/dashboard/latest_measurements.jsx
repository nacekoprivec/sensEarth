import { Row, Col, Card, Table, Spinner } from "react-bootstrap";

export default function LatestMeasurementsDashboard({ sensors, loading }) {

function parseAndFormatLocation(locationStr) {
  if (!locationStr) return "-";
  try {
    const loc = JSON.parse(locationStr);
    if (!loc.coordinates || loc.coordinates.length < 2) return "-";
    const [lon, lat, alt] = loc.coordinates;
    return (
      <a
        href={`https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}`}
        target="_blank"
        rel="noopener noreferrer"
      >
        {lat.toFixed(5)}, {lon.toFixed(5)}
        {alt ? ` (${alt.toFixed(1)} m)` : ""}
      </a>
    );
  } catch (err) {
    return "-";
  }
}
  return (
    <>
    <Card className="flat-card">
            <Card.Body>
              <div className="border-bottom d-flex justify-content-between align-items-center mb-3" > 
                <h3>Sensor Measurements</h3>
              </div>

              {loading ? (
                <div className="text-center">
                  <Spinner animation="border" />
                </div>
              ) : (
                <Table className="dashboard-table">
                  <thead>
                    <tr>
                      <th>Sensor</th>
                      <th>Timestamp</th>
                      <th>Location</th>
                      <th>Value</th>
                    </tr>
                  </thead>
    
                  <tbody>
                    {sensors.length === 0 ? (
                      <tr>
                        <td colSpan="4" className="text-center">
                          No measurements found
                        </td>
                      </tr>
                    ) : (
                      sensors.map((sensor, index) => (
                        <tr key={`${sensor.sensor_id}-${sensor.timestamp_utc}-${index}`}>
                          <td>
                            {sensor.sensor_label} ({sensor.sensor_id})
                          </td>
    
                          <td>
                            {new Date(sensor.timestamp_utc).toLocaleString()}
                          </td>
                          <td>{parseAndFormatLocation(sensor.location)}</td>            
                      <td>{sensor.value}</td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </Table>
              )}
            </Card.Body>
          </Card>
      </>
  );
}