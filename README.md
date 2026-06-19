# SensEarth

A digital twin of sensor data sources.

## Characteristics

SensEarth is designed to ingest, validate, enrich, model and serve sensor data from public and private sources. The key characteristics below describe the system goals and practical ideas for implementation.

- **Capture:** reliably collect data from heterogeneous open data sources (HTML pages, JSON APIs, CSVs, JS-rendered pages, streaming endpoints).
- **Data quality & validation:** apply automated quality checks (schema validation, completeness, ranges, duplicate detection, timestamp consistency) and track provenance for every raw payload.
- **Observability & monitoring:** monitor ingestion success/failure rates, lag, data volume, model performance, and alert on regressions.
- **Presentation:** provide dashboard summaries, time-series charts, and detailed records for individual sensors and sources.
- **Modularity & extensibility:** make source adapters, parsers, feature generators and models pluggable so adding a new source or model is low-effort.


## Architecture

### Scapers

Requirements:
* multiple scrapers can be implemented (they should be able to ingest various sources, like groundwarter levels, traffic data, web logs, weather)
* first, let's focus on scraping numerical data
* they should be robust; should not fail
* it should be easy to add an additional scraper
* scrapers can write directly to the database in the beginning (it is the easiest to implement)
* use something like scrapy in order to maintain robustness

### Storage System & Middleware

We use 2 storage systems for the scraped data: (1) raw data storage and (2) structured data storage.

**Raw Data Storage**:
* this is basically a document storage of all the parsed sources
* the raw scraped documents are stored here and can be later used to rebuild the database with historic data
* for implementation use MinIO; you can also use raw storage capabilities of postgres

**Structured Data Storage**:
* structured data storage will store all the data from the parsed sensors
* it will use the following tables:
    * sensor node (can host more sensors)
    * sensor (can belong to a node, or can be independent)
    * sensor type (defines phenomena, units, max/min values etc.)
    * sensor measurement
* for implementation use Postgres + TimescaleDB + PostGIS (optional)

**Middleware**:
* midleware routes data to modeling services as the data arrives
* middleware offers endpoints for retrieving the data for the user interface
* middleware enables registering of new sensors and linking them to scrapers

### Modelling

* we can have multiple models; such as time series prediction or anomaly detection or specialized services
* models are standalone and plug-and play


### Monitoring (WatchDog)

Requirements for monitoring:
* monitors if components of the system are running
* monitors success/failure of the scrapers
* monitors success/failure of the modeling services
* monitors timing of the models
* monitors uptime of the components


### User Interface

* use React + Vite (see `frontend/`)
* framework should display data from all the above components; monitoring, status of scrapers, data, models
* it should have a dashboard + possiblity to build custom dashboards
* it should have a detailed view of the data
* it should show error metrics of the models through time

## Architecture

High-level component view (all services run via Docker Compose):

```mermaid
flowchart TB
 subgraph DOCKER["Docker Compose"]
        SCR["Scrapers<br>web_scraper"]
        MINIO[("MinIO<br>raw storage")]
        MW["Middleware<br>FastAPI"]
        DB[("Core DB<br>PostgreSQL · TimescaleDB · PostGIS")]
        MODEL["Modeling<br>anomaly detection · forecasting"]
        MON["Monitoring<br>Monitoring API"]
        MONDB[("Monitoring DB<br>PostgreSQL")]
        UI["Web UI<br>Nginx · React · Vite · MapLibre"]
  end
    SRC["External sources<br>ARSO, Goriva.si, …"] --> SCR
    SCR --> MINIO & MW
    MW --> DB & MODEL
    MODEL --> DB
    MW -.-> MON
    SCR -.-> MON
    MODEL -.-> MON
    MON --> MONDB
    USER(["Browser"]) --> UI
    UI -- /middleware --> MW
    UI -- /monitoring --> MON
```

**Legend:** solid arrows = main data flow; dashed arrows = telemetry / monitoring and API proxy paths.

## How to Run

### Prerequisites
- Docker and Docker Compose installed on your system.

### Setup
1. Clone the repository:
   ```
   git clone <repository-url>
   cd sensEarth
   ```

2. Create a `.env` file in the root directory with the following environment variables or use .env.example:
   
   sensEarth/
   ```
   CORE_DB_URL=postgresql://postgres:postgres@localhost:5433/sensearth_db
   MONITORING_DB_URL=postgresql://postgres:postgres@localhost:5434/monitoring_db
   MIDDLEWARE_API=http://middleware-api:5006
   MONITORING_API=http://monitoring-api:8001

   ```
  
  sensEarth/frontend/
  ```
   VITE_MIDDLEWARE_API_URL=http://localhost:5006
   VITE_MONITORING_API_URL=http://localhost:8001
  ```
  
  In production, the frontend is configured to proxy API calls through Nginx at `/middleware` and `/monitoring`.
  The static build uses `frontend/.env.production` to route to the backend via the same origin.

3. Start the services:
   ```
   docker compose up
   ```
   This will build and start all containers, including databases, APIs, scrapers, and the frontend.

### Access Points
- **Frontend**: http://localhost:5005
- **Middleware API**: http://localhost:5006
- **Monitoring API**: http://localhost:8001
- **pgAdmin**: http://localhost:8082
- **MinIO Console**: http://localhost:9001 
- **Core DB**: localhost:5433
- **Monitoring DB**: localhost:5434

### Running Specific Services
- To run only certain services, use: `docker compose up <service-name>`
- For development, you can run services individually or use `docker compose up --build` to rebuild images.

### Scrapers
- Scrapers run automatically as part of the `scraper` service.
- To run historic imports or specific configs, use commands like:
  ```
  docker compose run scraper python scraper.py --historic
  docker compose run scraper python scraper.py --config arso_meteo
  ```

### Stopping
```
- Stop all services: `docker compose down`
- Remove volumes: `docker compose down -v`
  ```
