"""
Ingestion entry point: try Agmarknet first, fall back to local CSV,
then register the resulting raw dataset in the version registry.

This is the only function other modules should call to get raw data —
they should never import agmarknet_client or csv_loader directly.
"""

import logging

import pandas as pd

from src.config.settings import settings
from src.ingestion.agmarknet_client import fetch_all_target_commodities, AgmarknetFetchError
from src.ingestion.csv_loader import load_fallback_dataset, load_local_raw_files, CsvLoadError
from src.ingestion.dataset_registry import register_dataset_version

logger = logging.getLogger(__name__)


def ingest_raw_dataset() -> tuple[pd.DataFrame, dict]:
    """
    Returns (dataframe, manifest). Source priority:
      1. Agmarknet live API
      2. Local raw CSVs (data/raw/*.csv)
      3. Named fallback dataset

    Raises RuntimeError only if every source fails — ingestion should
    almost never hard-fail in a demo environment since the fallback
    CSV is checked into the repo.
    """
    source_used = None
    df = None

    try:
        df = fetch_all_target_commodities()
        source_used = "agmarknet_live"
    except AgmarknetFetchError as e:
        logger.warning("Agmarknet fetch failed, falling back: %s", e)

    if df is None:
        try:
            df = load_local_raw_files()
            source_used = "local_raw_csvs"
        except CsvLoadError as e:
            logger.warning("Local raw CSV load failed, trying named fallback: %s", e)

    if df is None:
        try:
            df = load_fallback_dataset()
            source_used = "named_fallback_csv"
        except CsvLoadError as e:
            raise RuntimeError(f"All ingestion sources exhausted: {e}")

    if len(df) < settings.min_rows_required:
        raise RuntimeError(
            f"Ingested dataset has only {len(df)} rows, "
            f"below minimum required ({settings.min_rows_required})"
        )

    manifest = register_dataset_version(
        df, source_name=source_used, stage="raw",
        extra_meta={"source_priority_used": source_used},
    )
    logger.info("Ingestion complete: %d rows from %s (hash=%s)",
                len(df), source_used, manifest["content_hash"][:12])

    return df, manifest