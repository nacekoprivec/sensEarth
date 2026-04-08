from psycopg2.extras import Json
import requests, threading
from typing import Optional


WATCHDOG_URL = "http://monitoring-api:8001"

def _send(path, payload):
    try:
        resp = requests.post(f"{WATCHDOG_URL}{path}", json=payload, timeout=5)
        print(f"[WatchDog client] Sent {path}, status: {resp.status_code}, response: {resp.text}")
    except Exception as e:
        print(f"[WatchDog client] Failed to send {path}: {e}")

def emit_heartbeat( name: str, instance_id: str, status: str = "OK", metadata: Optional[dict] = None):
    threading.Thread(
        target=_send,
        args=(
            "/heartbeat",
            {
                "name": name,
                "instance_id": instance_id,
                "status": status,
                "metadata": metadata or {},
            },
        ),
        daemon=True,
    ).start()
    
def emit_event(name: str, instance_id: str, event_type: str, severity: str = "INFO", message: Optional[str] = None, metadata: Optional[dict] = None):
    threading.Thread(
        target=_send,
        args=(
            "/event",
            {
                "name": name,
                "instance_id": instance_id,
                "event_type": event_type,
                "severity": severity,
                "message": message,
                "metadata": metadata or {},
            },
        ),
        daemon=True,
    ).start()

def emit_metric(name: str, instance_id: str, metric_name: str, value: float, unit: Optional[str] = None, metadata: Optional[dict] = None):
    threading.Thread(
        target=_send,
        args=(
            "/metric",
            {
                "name": name,
                "instance_id": instance_id,
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                "metadata": metadata or {},
            },
        ),
        daemon=True,
    ).start()

def emit_component_registration(name: str, instance_id: str, component_type: str = "other",):
    threading.Thread(
        target=_send,
        args=(
            "/component",
            {
                "name": name,
                "instance_id": instance_id,
                "type": component_type,
            },
        ),
        daemon=True,
    ).start()
