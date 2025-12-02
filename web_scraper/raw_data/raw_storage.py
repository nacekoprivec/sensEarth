from minio import Minio
from minio.error import S3Error
import io
import os

MINIO_ENDPOINT     = os.getenv("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY   = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY   = os.getenv("MINIO_SECRET_KEY", "minioadmin")
MINIO_BUCKET       = os.getenv("MINIO_BUCKET", "raw-data")

client = Minio(
    MINIO_ENDPOINT,
    access_key=MINIO_ACCESS_KEY,
    secret_key=MINIO_SECRET_KEY,
    secure=False,
)


def init_bucket() -> None:
    """Make sure the raw-data bucket exists."""

    if not client.bucket_exists(MINIO_BUCKET):
        client.make_bucket(MINIO_BUCKET)


def upload_raw_data(object_name: str, data: bytes, content_type="application/octet-stream") -> str:
    """Upload raw bytes to MinIO, return object path."""
    client.put_object(
        MINIO_BUCKET,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_name


def download_raw_data(object_name: str) -> bytes:
    """Download raw file from MinIO."""
    try:
        response = client.get_object(MINIO_BUCKET, object_name)
        data = response.read()
        response.close()
        response.release_conn()
        return data
    except S3Error:
        return None


def list_raw_objects(prefix: str = ""):
    """List objects inside the bucket."""
    return [obj.object_name for obj in client.list_objects(MINIO_BUCKET, prefix=prefix)]
