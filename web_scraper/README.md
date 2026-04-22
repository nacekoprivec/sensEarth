# Web Scrapers

This module fetches, extracts, and maps external data sources into the SensEarth format.

## Overview

The web scraper consists of:
- **Fetcher**: Retrieves raw data from URLs.
- **Extractor**: Parses data based on format (XML, JSON, CSV, HTML).
- **Mapper**: Transforms extracted data into nodes and sensors.
- **Enricher**: Adds metadata like coordinates.
- **Scraper**: Orchestrates the process, handles state, and sends data to the API.

## Configuration

Each scraper uses a JSON config file in `configs/`. Example:

```json
{
  "scraper_config": {
    "name": "GorivaSI_Diesel_Scraper",
    "description": "Scraper for diesel fuel prices from goriva.si API",
    "target_url": "https://goriva.si/api/v1/search/?format=json&position=Ljubljana",
    "fetch_interval": 1800,
    "format": "json",
    "root_tag": "results"
  },
  "mapping_config": {
    "node": {
      "node_label": "Fuel_Station",
      "node_serial": "pk",
      "longitude": "lng",
      "latitude": "lat",
      "altitude": null
    },
    "sensors": [
      {
        "sensor_label": "name",
        "sensor_name": "Diesel Price Sensor",
        "longitude": "lng",
        "latitude": "lat",
        "altitude": null,
        "sensor_description": "Diesel fuel price from goriva.si",
        "measurements": [
          {
            "value": "prices.dizel",
            "timestamp_utc": null
          }
        ],
        "sensor_type": {
          "name": "diesel_price",
          "phenomenon": "Diesel Fuel Price",
          "unit": "EUR/L",
          "value_min": 0,
          "value_max": 5
        }
      }
    ]
  }
}
```

### Required Parameters

- **scraper_config**:
  - `name`: Unique scraper identifier.
  - `description`: Optional description.
  - `target_url`: URL to scrape.
  - `fetch_interval`: Seconds between fetches (0 for one-time).
  - `format`: Data format (json, xml, html, csv).
  - `root_tag`: Root element or delimiter for extraction.

- **mapping_config**:
  - **node**: Defines node structure.
    - `node_label`: Node name.
    - `node_serial`: Unique node ID.
    - `longitude`, `latitude`: Required coordinates.
    - `altitude`: Optional.
  - **sensors**: Array of sensor definitions.
    - `sensor_label`: Label from scraped data.
    - `sensor_name`: Sensor name.
    - `longitude`, `latitude`, `altitude`: Coordinates (can reference node fields).
    - `sensor_description`: Optional.
    - `measurements`: Array of measurements.
      - `value`: Path to value (e.g., "prices.dizel").
      - `timestamp_utc`: Timestamp path or null (uses scrape time).
    - `sensor_type`: Required type definition.
      - `name`: Type name.
      - `phenomenon`: Description.
      - `unit`: Unit of measurement.
      - `value_min`, `value_max`: Value range.

## Usage

Run with Docker Compose:

- **Normal scraping**: `docker compose run scraper python scraper.py --config`
- **Historic import**: `docker compose run scraper python scraper.py --historic`
- **Replay MinIO data**: `docker compose run scraper python scraper.py --minio_reinsert`
- **Specific configs**: `docker compose run scraper python scraper.py --config config1 config2`

## Directory Structure

```
web_scraper/
|- configs/             # JSON scraper configurations
|- extractors/          # Format-specific extractors (XML, JSON, CSV, HTML)
|- fetcher.py           # Fetches raw data
|- enricher.py          # Cleanes and enriches data 
|- mapper.py            # Maps data to nodes/sensors
|- scraper.py           # Main scraper script
|- state/               # Persistent state files (auto-generated)
|- README.md
```

