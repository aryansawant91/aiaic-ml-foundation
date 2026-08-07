"""
Fallback / supplementary loader for local CSV datasets (e.g. the Kaggle
Indian Agriculture Crop Price Dataset, downloaded once into data/raw/).

Used when Agmarknet is unreachable or as a supplement to broaden
historical coverage. Column names are normalized to match the
Agmarknet schema so downstream code doesn't care which source a row
came from.
"""

import logging
from pathlib import Path

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

# Map common column name variants across public CSV sources to our
# canonical schema.
COLUMN_ALIASES = {
    "State": "state", "district_name": "district", "District": "district",
    "market_name": "market", "Market": "market",
    "Commodity": "commodity", "commodity_name": "commodity",
    "Variety": "variety",
    "Arrival_Date": "date", "date": "date", "Price Date": "date",
    "Min_Price": "min_price", "Max_Price": "max_price", "Modal_Price": "modal_price",
}


class CsvLoadError(Exception):
    pass


def load_fallback_dataset(filename: str = None) -> pd.DataFrame:
    filename = filename or settings.fallback_csv_name
    path = settings.raw_data_dir / filename

    if not path.exists():
        raise CsvLoadError(
            f"Fallback dataset not found at {path}. "
            f"Download it into data/raw/ before running ingestion."
        )

    df = pd.read_csv(path)
    df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})

    missing_required = {"commodity", "date"} - set(df.columns)
    if missing_required:
        raise CsvLoadError(f"Fallback dataset missing required columns: {missing_required}")

    logger.info("Loaded %d rows from fallback CSV %s", len(df), filename)
    return df


def load_local_raw_files() -> pd.DataFrame:
    """
    Load and concatenate every CSV in data/raw/ that matches the known
    schema. Useful once you've dropped multiple source files in manually.
    """
    frames = []
    for path in settings.raw_data_dir.glob("*.csv"):
        try:
            df = pd.read_csv(path)
            df = df.rename(columns={k: v for k, v in COLUMN_ALIASES.items() if k in df.columns})
            if "commodity" in df.columns and "date" in df.columns:
                df["_source_file"] = path.name
                frames.append(df)
        except Exception as e:
            logger.warning("Skipping unreadable file %s: %s", path, e)

    if not frames:
        raise CsvLoadError(f"No valid CSV files found in {settings.raw_data_dir}")

    return pd.concat(frames, ignore_index=True)