"""
Client for the data.gov.in Agmarknet "Variety-wise Daily Market Prices"
resource. This is the primary real-data source.

If no API key is set or the request fails, callers should fall back to
csv_loader.load_fallback_dataset() — this module never silently returns
fake data; it either returns real rows or raises.
"""

import logging
from typing import List, Dict, Any

import pandas as pd
import requests

from src.config.settings import settings

logger = logging.getLogger(__name__)

RENAME_MAP = {
    "state": "state",
    "district": "district",
    "market": "market",
    "commodity": "commodity",
    "variety": "variety",
    "arrival_date": "date",
    "min_price": "min_price",
    "max_price": "max_price",
    "modal_price": "modal_price",
}


class AgmarknetFetchError(Exception):
    pass


def _fetch_page(offset: int, limit: int, commodity: str, api_key: str) -> Dict[str, Any]:
    url = f"{settings.agmarknet_base_url}/{settings.agmarknet_resource_id}"
    params = {
        "api-key": api_key,
        "format": "json",
        "offset": offset,
        "limit": limit,
        "filters[commodity]": commodity,
    }
    response = requests.get(url, params=params, timeout=15)
    if response.status_code != 200:
        raise AgmarknetFetchError(
            f"Agmarknet request failed: {response.status_code} {response.text[:200]}"
        )
    return response.json()


def fetch_commodity_prices(commodity: str, max_records: int = 5000) -> pd.DataFrame:
    """
    Fetch up to max_records rows for a single commodity, paginating in
    batches of 1000 (the data.gov.in API's typical per-page cap).
    Raises AgmarknetFetchError if the API key is missing or the source
    is unreachable — the caller decides whether to fall back.
    """
    if not settings.agmarknet_api_key:
        raise AgmarknetFetchError("No DATA_GOV_IN_API_KEY configured")

    page_size = 1000
    all_records: List[Dict[str, Any]] = []
    offset = 0

    while offset < max_records:
        payload = _fetch_page(offset, page_size, commodity, settings.agmarknet_api_key)
        records = payload.get("records", [])
        if not records:
            break
        all_records.extend(records)
        offset += page_size
        if len(records) < page_size:
            break

    if not all_records:
        raise AgmarknetFetchError(f"No records returned for commodity={commodity}")

    df = pd.DataFrame(all_records)
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})
    logger.info("Fetched %d rows for commodity=%s", len(df), commodity)
    return df


def fetch_all_target_commodities() -> pd.DataFrame:
    """Fetch and concatenate data for every crop in settings.target_crops."""
    frames = []
    errors = []
    for crop in settings.target_crops:
        try:
            frames.append(fetch_commodity_prices(crop))
        except AgmarknetFetchError as e:
            logger.warning("Skipping %s: %s", crop, e)
            errors.append(str(e))

    if not frames:
        raise AgmarknetFetchError(f"All commodity fetches failed: {errors}")

    return pd.concat(frames, ignore_index=True)