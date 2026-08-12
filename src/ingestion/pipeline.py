"""
Ingestion entry point: load from local CSVs in data/raw/, with named
fallback CSV as backup. Agmarknet removed — real Kaggle mandi datasets
used instead (no API key required, real government-sourced price data).
"""

import logging
import pandas as pd

from src.ingestion.csv_loader import load_local_raw_files, load_fallback_dataset, CsvLoadError
from src.ingestion.dataset_registry import register_dataset_version
from src.config.settings import settings

logger = logging.getLogger(__name__)


def ingest_raw_dataset() -> tuple[pd.DataFrame, dict]:
    """
    Source priority:
      1. All CSVs in data/raw/ (picks up both datasets if both are dropped in)
      2. Named fallback CSV (single file)

    Raises RuntimeError only if both sources fail — which won't happen
    as long as at least one CSV exists in data/raw/.
    """
    source_used = None
    df = None

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
            f"below minimum ({settings.min_rows_required})"
        )

    manifest = register_dataset_version(
        df, source_name=source_used, stage="raw",
        extra_meta={"source_priority_used": source_used},
    )
    logger.info(
        "Ingestion complete: %d rows from %s (hash=%s)",
        len(df), source_used, manifest["content_hash"][:12],
    )
    return df, manifest