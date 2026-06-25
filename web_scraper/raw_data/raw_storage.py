from minio import Minio
from minio.error import S3Error
import io
import os
import time

from monitoring.client import emit_event, emit_metric, emit_heartbeat
from utils import safe_emit

MINIO_ENDPOINT     = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET       = os.getenv("MINIO_BUCKET", "raw-data")
MINIO_INSTANCE_ID  = "default"

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)

def init_bucket(retries=10, delay=2):
    for i in range(retries):
        try:
            if not client.bucket_exists(MINIO_BUCKET):
                client.make_bucket(MINIO_BUCKET)
            print("[MinIO] Bucket ready")
            return
        except Exception as e:
            print(f"[MinIO] Waiting... ({i+1}/{retries})")
            time.sleep(delay)

    safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="FAIL")
    safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="bucket_init_failed", severity="ERROR", message=f"MinIO not available after {retries} retries")
    raise RuntimeError("MinIO not available")

def upload_raw_data(object_name: str, data: bytes, content_type="application/octet-stream") -> str:
    """Upload raw bytes to MinIO, return object path."""
    start = time.time()
    try:
        client.put_object(
            MINIO_BUCKET,
            object_name,
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        duration = time.time() - start
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="OK")
        safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="upload_success", severity="INFO", message=f"Uploaded {object_name}", metadata={"object_name": object_name})
        safe_emit(emit_metric, name="minio", instance_id=MINIO_INSTANCE_ID, metric_name="upload_bytes", value=len(data), unit="bytes")
        safe_emit(emit_metric, name="minio", instance_id=MINIO_INSTANCE_ID, metric_name="upload_duration_seconds", value=duration, unit="seconds")
        return object_name
    except S3Error as e:
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="FAIL")
        safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="upload_failed", severity="ERROR", message=f"Failed to upload {object_name}: {e}", metadata={"object_name": object_name})
        raise

def download_raw_data(object_name: str) -> bytes:
    """Download raw file from MinIO."""
    start = time.time()
    try:
        response = client.get_object(MINIO_BUCKET, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="OK")
        safe_emit(emit_metric, name="minio", instance_id=MINIO_INSTANCE_ID, metric_name="download_bytes", value=len(data), unit="bytes")
        safe_emit(emit_metric, name="minio", instance_id=MINIO_INSTANCE_ID, metric_name="download_duration_seconds", value=time.time() - start, unit="seconds")
        return data
    except S3Error as e:
        print(f"[MinIO] Error downloading {object_name}: {e}")
        safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="download_failed", severity="ERROR", message=f"Failed to download {object_name}: {e}", metadata={"object_name": object_name})
        return None


def list_raw_objects(prefix: str = ""):
    """List objects inside the bucket."""
    try:
        objects = [obj.object_name for obj in client.list_objects(MINIO_BUCKET, prefix=prefix, recursive=True)]
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="OK")
        safe_emit(emit_metric, name="minio", instance_id=MINIO_INSTANCE_ID, metric_name="object_count", value=len(objects), unit="count")
        return objects
    except S3Error as e:
        safe_emit(emit_heartbeat, name="minio", instance_id=MINIO_INSTANCE_ID, status="FAIL")
        safe_emit(emit_event, name="minio", instance_id=MINIO_INSTANCE_ID, event_type="list_failed", severity="ERROR", message=f"Failed to list objects with prefix '{prefix}': {e}")
        raise

def object_exists(object_name: str) -> bool:
    try:
        client.stat_object(MINIO_BUCKET, object_name)
        return True
    except S3Error:
        return False