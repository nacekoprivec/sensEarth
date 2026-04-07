import requests
from storage_pipeline import store_raw_response 

class Fetcher:
    """Basic fetcher using requests."""
    def fetch(self, url: str) -> bytes:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            object_name, is_new = store_raw_response(response) 

            return {
                "content": response.content,
                "object_name": object_name,
                "is_new": is_new
            }
        except requests.RequestException as e:
            print(f"[Fetcher] Error fetching {url}: {e}")
            return {
                "content": b"",
                "object_name": None,
                "is_new": False
            }
