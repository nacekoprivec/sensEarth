import time
from requests.exceptions import RequestException
from datetime import datetime
import os
import json


def load_configs(folder="configs", selected=None, include_csv=False):
    """
    Loads every .json config pair from the given folder.

    include_csv controls format policy:
      - False (default): skip CSV configs — used by the continuous scraper.
      - True: keep only CSV configs — used by historic import.
    """
    configs = []
    folder_path = os.path.join(os.path.dirname(__file__), folder)
    for file in os.listdir(folder_path):
        if not file.endswith(".json"):
            continue
        name = os.path.splitext(file)[0]

        if selected and name not in selected:
            continue
        with open(os.path.join(folder_path, file), "r") as f:
            data = json.load(f)

        format_type = data.get("scraper_config", {}).get("format", "").lower()
        is_csv = format_type == "csv"
        # Continuous mode wants non-CSV; historic mode wants CSV only.
        if is_csv != include_csv:
            continue
        configs.append((data["scraper_config"], data["mapping_config"]))

    return configs
 

def retry_request(func, retries=5, delay=5, backoff=2, *args, **kwargs):
    """
    Retry a request function multiple times with exponential backoff.
    func: callable (like requests.post)
    retries: number of attempts
    delay: initial delay in seconds
    backoff: multiply delay each retry
    *args, **kwargs: passed to func
    """
    current_delay = delay
    for attempt in range(1, retries + 1):
        try:
            return func(*args, **kwargs)
        except RequestException as e:
            print(f"Attempt {attempt} failed: {e}. Retrying in {current_delay}s...")
            time.sleep(current_delay)
            current_delay *= backoff
    raise ConnectionError(f"Failed after {retries} attempts")


def request_with_retry(method, url, *, retries=5, delay=3, backoff=2, **kwargs):
    """
    HTTP call that retries on connection AND HTTP 5xx errors.

    raise_for_status runs inside the retried function so a 5xx response is
    retried (HTTPError is a RequestException), not only connection errors.
    """
    def _call():
        response = method(url, **kwargs)
        response.raise_for_status()
        return response

    return retry_request(_call, retries=retries, delay=delay, backoff=backoff)

def safe_emit(func, **kwargs):
    try:
        func(**kwargs)
    except Exception:
        pass

def normalize(payload: dict):
    """
    Registration-time fixes for fields that may appear on node/sensor
    entries in the /register payload but outside mapped measurements.

    """
    for entity_type in ("nodes", "sensors"):
        for item in payload.get(entity_type, []):
            if "timestamp_utc" not in item:
                continue
            ts = item.get("timestamp_utc")
            if ts is None or (isinstance(ts, str) and ts.strip().lower() in ("", "null")):
                item["timestamp_utc"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def normalize_timestamp(ts: str) -> str:
    if ts is None or ts.strip() == "" or ts.lower().strip() == "null":
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    ts = ts.strip().replace("UTC", "").strip()

    formats = [
        "%d.%m.%Y",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
    ]

    if ts == "" or ts.lower() == "null":
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    for fmt in formats:
        try:
            dt = datetime.strptime(ts, fmt)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue

    raise ValueError(f"Unrecognized timestamp format: {ts}")