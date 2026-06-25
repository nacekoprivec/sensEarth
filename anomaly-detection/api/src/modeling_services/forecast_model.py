from .base import BaseModel
from src.DetectionPipeline import DetectionPipeline
from monitoring.client import emit_component_registration, emit_heartbeat, emit_event, emit_metric

class ForecastModel(BaseModel):
    def __init__(self, model_name, sensor_id, conf, sliding_window_size=100):
        super().__init__(model_name, sensor_id, conf, sliding_window_size)

        emit_heartbeat(name="forecast_model", instance_id=self.model_name, status="INITIALIZED")

    def data_ingestion(self, measurements):
        self.data = []
        self.data = [
        {
            "timestamp": float(m.timestamp_utc.timestamp()),
            "ftr_vector": [float(m.value)],
        }
        for m in measurements
        ]

        self.model_run = DetectionPipeline(conf=self.conf)

        emit_metric(
            name="forecast_model",
            instance_id=self.model_name,
            metric_name="ingested_measurements",
            value=len(self.data),
            unit="count"
        )

        emit_event(
            name="forecast_model",
            instance_id=self.model_name,
            event_type="data_ingested",
            severity="INFO",
            message=f"Ingested {len(self.data)} measurements"
        )

    def run(self):
        emit_heartbeat(name="forecast_model", instance_id=self.model_name, status="OK")

        try:
            self.result = self.model_run.read_streaming_data(self.data)    
            #for r in self.result:
                #print(f"Measurement: {r[0]}, Forecast Result: {r}")    # Measurement: [20536.6042, 104.0], Forecast Result: ([20536.6042, 104.0], ('Error: Value over the upper limit', -1, '72.90424168216992'))
                
            emit_metric(
                name="forecast_model",
                instance_id=self.model_name,
                metric_name="processed_measurements",
                value=len(self.data),
                unit="count"
            )

            emit_event(
                name="forecast_model",
                instance_id=self.model_name,
                event_type="processing_completed",
                severity="INFO",
                message=f"Processed {len(self.data)} measurements"
            )

            emit_heartbeat(name="forecast_model", instance_id=self.model_name, status="OK")

            return self.result

        except Exception as e:
            emit_heartbeat(
                name="forecast_model",
                instance_id=self.model_name,
                status="FAIL"
            )

            emit_event(
                name="forecast_model",
                instance_id=self.model_name,
                event_type="processing_failed",
                severity="ERROR",
                message=str(e)
            )

            raise

    def fetch_data(self):
        pass
