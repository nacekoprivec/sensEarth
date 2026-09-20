"""
Fetch ARSO pov_arhiv daily CSV for a station sifra / year range.
"""

from __future__ import annotations

from typing import Optional
from urllib.parse import urlencode

import requests

from utils import request_with_retry

DAILY_CSV_BASE = "https://vode.arso.gov.si/hidarhiv/pov_arhiv_tab.php"


def build_daily_csv_url(
    sifra: str,
    year_from: int,
    year_to: int,
) -> str:
    query = urlencode(
        {
            "p_postaja": str(sifra),
            "p_od_leto": str(year_from),
            "p_do_leto": str(year_to),
            "b_oddo_CSV": "Izvoz dnevnih vrednosti v CSV",
        }
    )
    return f"{DAILY_CSV_BASE}?{query}"


def fetch_daily_csv(
    sifra: str,
    year_from: int,
    year_to: int,
    timeout: int = 60,
) -> bytes:
    url = build_daily_csv_url(sifra, year_from, year_to)
    # request_with_retry runs raise_for_status inside the retry so 5xx from the
    # ARSO archive is retried, not only connection errors.
    response = request_with_retry(requests.get, url, timeout=timeout)
    return response.content
