from raw_data.raw_storage import upload_raw_data, init_bucket
import datetime
import hashlib
import json


init_bucket()   # ensure bucket exists once on import


def generate_object_name(prefix: str, url: str, ext: str) -> str:
    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    h = hashlib.md5(url.encode()).hexdigest()[:10]
    return f"{prefix}/{ts}_{h}.{ext}"


def store_raw_response(response):
    """Save raw HTTP body to MinIO."""
    object_name = generate_object_name("raw", response.url, "bin")
    upload_raw_data(
        object_name,
        response.body,
        content_type=response.headers.get("Content-Type", b"application/octet-stream").decode()
    )
    return object_name
