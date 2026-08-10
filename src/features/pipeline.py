"""
Feature engineering orchestrator: season features -> lag features ->
rolling features -> version registration.

Order matters: season features don't depend on sort order, so they're
added first for clarity. Lag and rolling both require the frame to be
sorted by (commodity, state, date), which each function guarantees
internally, but running lag before rolling avoids redundant sorts.
"""

import logging

import pandas as pd

from src.features.season_encoding import add_season_features
from src.features.lag_features import add_lag_features
from src.features.rolling_features import add_rolling_features
from src.ingestion.dataset_registry import register_dataset_version

logger = logging.getLogger(__name__)

# Columns that become unusable (NaN) for the earliest rows of each
# commodity/state series once lag/rolling features are added.
FEATURE_DERIVED_NA_COLUMNS_PREFIX = ("modal_price_lag_", "modal_price_roll_")


def run_feature_engineering(df: pd.DataFrame, parent_hash: str) -> tuple[pd.DataFrame, dict]:
    working = add_season_features(df)
    working = add_lag_features(working)
    working = add_rolling_features(working)

    derived_cols = [
        c for c in working.columns
        if c.startswith(FEATURE_DERIVED_NA_COLUMNS_PREFIX)
    ]
    before = len(working)
    working = working.dropna(subset=derived_cols).reset_index(drop=True)
    dropped = before - len(working)
    logger.info(
        "Feature engineering: dropped %d rows lacking full lag/rolling history "
        "(expected for series start), %d rows remain",
        dropped, len(working),
    )

    manifest = register_dataset_version(
        working, source_name="feature_pipeline", stage="features",
        parent_hash=parent_hash,
        extra_meta={"rows_dropped_insufficient_history": dropped, "feature_columns": derived_cols},
    )
    logger.info(
        "Feature engineering complete: %d rows -> hash=%s",
        len(working), manifest["content_hash"][:12],
    )
    return working, manifest