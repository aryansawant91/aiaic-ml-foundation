"""
Preprocessing orchestrator: missing values -> outliers -> version
registration. Sits between validation and feature engineering.
"""

import logging

import pandas as pd

from src.preprocessing.missing_values import handle_missing_values
from src.preprocessing.outliers import handle_outliers
from src.ingestion.dataset_registry import register_dataset_version

logger = logging.getLogger(__name__)


def run_preprocessing(df: pd.DataFrame, parent_hash: str) -> tuple[pd.DataFrame, dict]:
    """
    Applies missing value handling then outlier capping, in that fixed
    order (outlier detection is more reliable once nulls are resolved,
    since groupby medians would otherwise skip null-containing rows
    inconsistently).
    """
    working = handle_missing_values(df)
    working = handle_outliers(working)

    manifest = register_dataset_version(
        working, source_name="preprocessing_pipeline", stage="processed",
        parent_hash=parent_hash,
    )
    logger.info(
        "Preprocessing complete: %d rows -> hash=%s",
        len(working), manifest["content_hash"][:12],
    )
    return working, manifest