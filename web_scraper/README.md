# web_scraper


## Examples

Each config file should look like this:

```json
{
  "name": "detector1",
  "config_name": "border_check.json",
  "description": "default description",
  "target_url": "http://hmljn.arso.gov.si/vode/podatki/stanje_voda_samodejne.html",
  "selector": "table.online",
  "fetch_interval": 1,
  "limit_results": 1
}

## Run Minio 

docker compose up -d

Go to http://localhost:9000

## Run scrapers

activate env
conda activate web_scraper_env

All commands are run from the project root:
python custom_scraper.py [OPTIONS] 

Run every scraper from configs
python custom_scraper.py --config

Run only specific configs
python custom_scraper.py --config detector1 detector2

Test only specific detector 
python custom_scraper.py --detect config1 -ftr_vector 13 -timestamp 14



{
  "name": "ARSO Water Levels",
  "description": "Scraper for water level data from ARSO",
  "target_url": "http://hmljn.arso.gov.si/vode/podatki/stanje_voda_samodejne.html",
  "fetch_interval": 1,
  "selector": "None",
  
  "sensor_type": "water_level",
  "sensor_node": "None",
  "unit": "cm"
}