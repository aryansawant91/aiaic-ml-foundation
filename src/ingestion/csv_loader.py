"""
Fallback / supplementary loader for local CSV datasets.
Column names normalized to canonical schema regardless of source formatting.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

COLUMN_ALIASES = {
    # State variants
    "State": "state", "STATE": "state", "state_name": "state",

    # District variants
    "District Name": "district", "district_name": "district",
    "District": "district", "DISTRICT": "district",

    # Market variants
    "Market Name": "market", "market_name": "market",
    "Market": "market", "MARKET": "market",

    # Commodity variants
    "Commodity": "commodity", "COMMODITY": "commodity",
    "commodity_name": "commodity", "Commodity Name": "commodity",

    # Variety variants
    "Variety": "variety", "VARIETY": "variety",

    # Date variants
    "Arrival_Date": "date", "arrival_date": "date",
    "Price Date": "date", "DATE": "date", "Date": "date",

    # Price variants — normal
    "Min_Price": "min_price", "MIN_PRICE": "min_price",
    "Max_Price": "max_price", "MAX_PRICE": "max_price",
    "Modal_Price": "modal_price", "MODAL_PRICE": "modal_price",

    # Price variants — URL-encoded spaces (x0020 = space in some exports)
    "Min_x0020_Price": "min_price",
    "Max_x0020_Price": "max_price",
    "Modal_x0020_Price": "modal_price",

    # Grade (not in schema but keep it, don't break on it)
    "Grade": "grade", "GRADE": "grade",
}

# After aliasing, if BOTH an original and alias column exist (e.g.
# "state" from alias AND "STATE" original both present), keep only
# the canonical one and drop the original.
CANONICAL_COLUMNS = {"state", "district", "market", "commodity", "variety",
                     "date", "min_price", "max_price", "modal_price"}


class CsvLoadError(Exception):
    pass


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
    # Drop leftover original-cased duplicates if canonical already exists
    drop_cols = [c for c in df.columns if c not in CANONICAL_COLUMNS
                 and c not in ("grade", "_source_file")]
    if drop_cols:
        logger.info("Dropping unrecognized/duplicate columns after normalization: %s", drop_cols)
        df = df.drop(columns=drop_cols)
    return df


def load_fallback_dataset(filename: str = None) -> pd.DataFrame:
    filename = filename or settings.fallback_csv_name
    path = settings.raw_data_dir / filename

    if not path.exists():
        raise CsvLoadError(
            f"Fallback dataset not found at {path}. "
            f"Download it into data/raw/ before running ingestion."
        )

    df = pd.read_csv(path, low_memory=False)
    df = _normalize(df)

    missing_required = {"commodity", "date"} - set(df.columns)
    if missing_required:
        raise CsvLoadError(
            f"Fallback dataset missing required columns after normalization: {missing_required}. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Loaded %d rows from fallback CSV %s", len(df), filename)
    return df


def load_local_raw_files() -> pd.DataFrame:
    frames = []
    for path in settings.raw_data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path, low_memory=False)
            df = _normalize(df)
            if "commodity" in df.columns and "date" in df.columns:
                df["_source_file"] = path.name
                frames.append(df)
                logger.info(
                    "Loaded %s: %d rows, columns=%s",
                    path.name, len(df), list(df.columns)
                )
            else:
                logger.warning(
                    "Skipping %s: missing commodity/date after normalization. "
                    "Columns found: %s", path.name, list(df.columns)
                )
        except Exception as e:
            logger.warning("Skipping unreadable file %s: %s", path, e)

    if not frames:
        raise CsvLoadError(f"No valid CSV files found in {settings.raw_data_dir}")

    return pd.concat(frames, ignore_index=True)