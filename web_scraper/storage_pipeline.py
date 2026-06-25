from raw_data.raw_storage import upload_raw_data, init_bucket, object_exists
from minio.error import S3Error
import datetime
import hashlib
import json

from monitoring.client import emit_event, emit_metric
from utils import safe_emit

init_bucket()   # ensure bucket exists once on import

def calculate_content_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def generate_object_name(content_hash: str, ext="bin") -> str:
    return f"raw/{content_hash}.{ext}"

def store_raw_response(response):
    content = response.content

    # Hash raw document
    content_hash = calculate_content_hash(content)

    # Generate deterministic object name
    ext = response.headers.get("Content-Type", "application/octet-stream").split("/")[-1]
    object_name = generate_object_name(content_hash, ext)

    # Skip upload if already exists
    if object_exists(object_name):
        print(f"[Storage] Duplicate skipped: {object_name}")
        safe_emit(emit_event, name="minio", instance_id="default", event_type="duplicate_skipped", severity="INFO", message=f"Skipped duplicate object {object_name}", metadata={"object_name": object_name})
        safe_emit(emit_metric, name="minio", instance_id="default", metric_name="duplicate_skip_count", value=1, unit="count")
        return object_name, False

    # Upload to MinIO
    upload_raw_data(
        object_name,
        content,
        content_type=response.headers.get(
            "Content-Type", "application/octet-stream"
        )
    )

    print(f"[Storage] Stored new object: {object_name}")
    return object_name, True
