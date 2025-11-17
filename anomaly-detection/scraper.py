import time
import pandas as pd

url = "http://hmljn.arso.gov.si/vode/podatki/stanje_voda_samodejne.html"
tables = pd.read_html(url)

# The water table is usually index 2
df = tables[2]


# Combine the multi-level header into single row

#print first ID 0                 Mura      Gornja Radgona     74.0   79.700   pada        19.2 , just the first 0
vodostaj = df["Vodostaj", "cm"].head(3)   # first 5 rows

# Get current Unix timestamp (float)
timestamp = float(time.time())

# Build structured datapoints
datapoints = []
for idx, value in enumerate(vodostaj):
    datapoints.append({
        "place_id": idx+1,       # 0,1,2,3,4
        "timestamp": timestamp,
        "vodostaj": float(value) if pd.notna(value) else None
    })

print(datapoints)


