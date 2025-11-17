import sys
import asyncio

if sys.platform.startswith("win") and sys.version_info >= (3, 8):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from twisted.internet import asyncioreactor
try:
    asyncioreactor.install()  # ensures Twisted uses AsyncioSelectorReactor not SelectorReactor
except Exception:
    pass

import requests
from crochet import setup, wait_for

setup()  # initialize Crochet AFTER installing reactor

import scrapy
from scrapy.crawler import CrawlerRunner
from scrapy.http import HtmlResponse
import traceback
import random
import datetime
import matplotlib.pyplot as plt
import streamlit as st
import json
import os
import argparse
import threading


API_URL = "http://127.0.0.1:8000"

setup()  # initializes Twisted reactor once globally

@wait_for(timeout=30.0)
def run_scrapy_sync(scraper):
    """Run Scrapy synchronously with Crochet (safe across threads)"""
    runner = CrawlerRunner(settings={"LOG_LEVEL": "ERROR"})
    return runner.crawl(
        ScrapySpider,
        target_url=scraper.target_url,
        selector=scraper.selector,
        limit_results=scraper.limit_results
    )

class ScrapySpider(scrapy.Spider):
    name = "scrapy_spider"
    result_data = []

    def __init__(self, target_url, selector=None, limit_results=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_urls = [target_url]
        self.selector = selector
        self.limit_results = limit_results
        self.data_format = None  # auto-detect based on URL or response

    def parse(self, response):
        content_type = response.headers.get("Content-Type", b"text/html").decode().lower()
        url_lower = response.url.lower()

        # Detect format from Content-Type or URL
        if "json" in content_type or url_lower.endswith(".json"):
            self.data_format = "json"
            return self.parse_json(response)
        elif "csv" in content_type or url_lower.endswith(".csv"):
            self.data_format = "csv"
            return self.parse_csv(response)
        else:
            self.data_format = "html"
            return self.parse_html(response)

    def parse_html(self, response: HtmlResponse):
        element = response.css(self.selector)
        if not element:
            self.logger.warning(f"No element found with selector '{self.selector}'")
            return
        rows = element.css("tr")
        if self.limit_results:
            rows = rows[2:self.limit_results + 2]
        for row in rows:
            ScrapySpider.result_data.append(list(row.css("::text").getall()))

    def parse_json(self, response):
        try:
            data = json.loads(response.text)
            if isinstance(data, dict):
                data = [data]
            if self.limit_results:
                data = data[:self.limit_results]
            for item in data:
                ScrapySpider.result_data.append(item)
        except Exception as e:
            self.logger.error(f"JSON parse error: {e}")

    def parse_csv(self, response):
        try:
            import csv
            from io import StringIO

            f = StringIO(response.text)
            reader = csv.DictReader(f)
            count = 0
            for row in reader:
                ScrapySpider.result_data.append(row)
                count += 1
                if self.limit_results and count >= self.limit_results:
                    break
        except Exception as e:
            self.logger.error(f"CSV parse error: {e}")

def fetch_data(self):
    """Fetch data using Scrapy spider in a separate thread."""
    try:
        ScrapySpider.result_data = []  # clear previous results
        thread = threading.Thread(target=run_scrapy_sync, args=(self,))
        thread.start()
        thread.join()
        return ScrapySpider.result_data
    except Exception as e:
        print(f"[{self.name}] Scrapy fetch error: {e}")
        return []

class Scraper:
    def __init__(self, name, config_name, description, target_url, selector,
                 fetch_interval=0, limit_results=1, api_url=API_URL):
        self.name = name
        self.config_name = config_name
        self.description = description
        self.target_url = target_url
        self.selector = selector
        self.fetch_interval = fetch_interval * 60
        self.limit_results = limit_results
        self.api_url = api_url
        self.detector_id = None

    def fetch_data(self):
        """Fetch data using Scrapy spider in a separate thread."""
        try:
            thread = threading.Thread(target=run_scrapy_sync, args=(self,))
            thread.start()
            thread.join()
            return ScrapySpider.result_data
        except Exception as e:
            print(f"[{self.name}] Scrapy fetch error: {e}")
            return []

    def get_detectors(self):
        """Fetch all anomaly detectors from FastAPI backend"""
        try:
            response = requests.get(f"{self.api_url}/detectors")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching detectors: {e}")
            return []

    def create_detector(self):
        """Create a new detector if it doesn't already exist, and set self.detector_id."""
        try:
            detectors = self.get_detectors()

            # Check if detector with same name exists
            existing = [d for d in detectors if d.get("name") == self.name]
            if existing:
                detector = existing[0]
                self.detector_id = detector.get("id")
                print(f"Detector '{self.name}' already exists with ID {self.detector_id}.")
                self.set_detector_status("active")
                return detector

            # Determine next detector_id
            if detectors:
                max_id = max(d.get("id", 0) for d in detectors)
                self.detector_id = max_id + 1
            else:
                self.detector_id = 1

            payload = {
                "id": self.detector_id,
                "name": self.name,
                "description": self.description,
                "config_name": self.config_name,
            }

            print(f"Creating new detector '{self.name}' with ID {self.detector_id}...")
            response = requests.post(f"{self.api_url}/detectors/create", json=payload)
            response.raise_for_status()
            print(f"Created new detector '{self.name}' successfully.")
            self.set_detector_status("active")
            return response.json()

        except Exception as e:
            print(f"Error creating detector {self.name}: {e}")
            return None
        
    def set_detector_status(self, status: str):
        """Set detector status to 'active' or 'inactive' via FastAPI endpoint."""
        try:
            url = f"{self.api_url}/detectors/{self.detector_id}/{status}"
            response = requests.put(url)
            response.raise_for_status()
            print(f"[{self.name}] Detector {self.detector_id} status set to '{status}'")
            return response.json()
        except requests.RequestException as e:
            print(f"[{self.name}] Error setting status to '{status}': {e}")
            return None
    
    def delete_detector(self):
        """Delete the detector via FastAPI endpoint."""
        try:
            url = f"{self.api_url}/detectors/{self.detector_id}/delete"
            response = requests.delete(url)
            response.raise_for_status()
            print(f"[{self.name}] Detector {self.detector_id} deleted successfully.")
            return response.json()
        except requests.RequestException as e:
            print(f"[{self.name}] Error deleting detector: {e}")
            return None

    def is_anomaly(self, timestamp: str, ftr_vector: float, detector_id: int = 1):
        """Check if provided feature vector is an anomaly."""
        try:
            url = f"{self.api_url}/detectors/{self.detector_id}/detect_anomaly"
            params = {"timestamp": timestamp, "ftr_vector": ftr_vector}
            response = requests.post(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"[{self.name}] Error calling anomaly API: {e}")
            return None

    async def run(self):
        """Continuously fetch data every fetch_interval seconds."""
        while True:
            data = self.fetch_data()
            if data:
                print(f"[{self.name}] Data: {data}")
                result = self.is_anomaly(str(data[0][2]), float(data[0][3]))
                print(f"[{self.name}] Anomaly result:", result)
            if self.fetch_interval == 0:
                break
            await asyncio.sleep(self.fetch_interval)

def load_configs(selected=None):
    """Load all or specific configs from /configs folder."""
    configs = []
    folder = os.path.join(os.path.dirname(__file__), "configs")

    for file in os.listdir(folder):
        if file.endswith(".json"):
            name = os.path.splitext(file)[0]
            if selected and name not in selected:
                continue
            with open(os.path.join(folder, file), "r") as f:
                conf = json.load(f)
                conf["name"] = name
                configs.append(conf)
    return configs

def delete_detector_by_id(detector_id: int, api_url=API_URL):
    """Delete detector by ID via FastAPI endpoint."""
    try:
        url = f"{api_url}/detectors/{detector_id}"
        response = requests.delete(url)
        response.raise_for_status()
        print(f"Detector {detector_id} deleted successfully.")
        return response.json()
    except requests.RequestException as e:
        print(f"Error deleting detector {detector_id}: {e}")
        return None
    

async def main():
    parser = argparse.ArgumentParser(description="Anomaly Detector CLI")
    parser.add_argument("--config", nargs="*", help="Specify which config(s) to use (none = all)")
    parser.add_argument("--detect", type=str, help="Run anomaly detection for given detector name")
    parser.add_argument("-ftr_vector", type=float, help="Feature vector value for manual detection")
    parser.add_argument("-timestamp", type=str, help="Timestamp for manual detection")
    parser.add_argument("--delete", type=str, help="Detector id for manual deletion")
    parser.add_argument("--visualization", type=str, help="Detector id for visualization (not implemented)")
    args = parser.parse_args()

    # Load config(s)
    if args.config is not None:
        configs = load_configs(selected=args.config if args.config else None)

        if args.detect is None:
            # Initialize or create detectors from config
            scrapers = [Scraper(**cfg) for cfg in configs]
            for scraper in scrapers:
                scraper.create_detector()
            await asyncio.gather(*(scraper.run() for scraper in scrapers))
            return

    # Manual anomaly detection
    if args.detect:
        configs = load_configs([args.detect])
        if not configs:
            print(f"No config found for detector '{args.detect}'")
            sys.exit(1)
        scraper = Scraper(**configs[0])
        scraper.create_detector()
        result = scraper.is_anomaly(args.timestamp, args.ftr_vector)
        print(f"Manual anomaly detection for '{args.detect}':", result)
    
    # Manual deletion
    if args.delete:
        delete_detector_by_id(int(args.delete))
        print(f"Deleted detector with ID {args.delete}")
        return 0
    
    if args.visualization:
        print("Visualization not implemented yet.")
        return 0

if __name__ == "__main__":
    asyncio.run(main())
