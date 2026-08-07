"""
Deterministic missing-value handling for the validated dataset.

Strategy is explicit and logged per column so "why was this value
filled this way" is always answerable — a hidden imputation choice
is the easiest way to break replay-safety.
"""

import logging
from typing import Dict

import pandas as pd

logger = logging.getLogger(__name__)

# Columns filled with group-wise median (numeric, price-like)
MEDIAN_FILL_COLUMNS = ["min_price", "max_price"]

# Columns filled with a constant placeholder (categorical, low-cardinality)
CONSTANT_FILL_VALUES: Dict[str, str] = {
    "district": "UNKNOWN_DISTRICT",
    "market": "UNKNOWN_MARKET",
    "variety": "UNSPECIFIED",
}


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing values using a fixed, documented strategy:

      - min_price / max_price: median within (commodity, state) group,
        falling back to global median if the group itself is all-null.
      - district / market / variety: constant placeholder string.
      - modal_price: never filled — rows missing this should already
        have been quarantined by validation; if any slip through we
        drop them here rather than guess a target value.

    Returns a new dataframe; does not mutate the input.
    """
    working = df.copy()
    fill_report: Dict[str, int] = {}

    # Never impute the prediction target.
    if "modal_price" in working.columns:
        before = len(working)
        working = working[working["modal_price"].notna()].reset_index(drop=True)
        dropped = before - len(working)
        if dropped:
            logger.warning("Dropped %d rows with null modal_price (target column)", dropped)
            fill_report["modal_price_dropped_rows"] = dropped

    for col in MEDIAN_FILL_COLUMNS:
        if col not in working.columns:
            continue
        null_count = int(working[col].isna().sum())
        if null_count == 0:
            continue

        group_median = working.groupby(["commodity", "state"])[col].transform("median")
        working[col] = working[col].fillna(group_median)

        # Remaining nulls (whole group was null) -> global median
        remaining_null = working[col].isna()
        if remaining_null.any():
            global_median = working[col].median()
            working.loc[remaining_null, col] = global_median

        fill_report[col] = null_count

    for col, placeholder in CONSTANT_FILL_VALUES.items():
        if col not in working.columns:
            continue
        null_count = int(working[col].isna().sum())
        if null_count == 0:
            continue
        working[col] = working[col].fillna(placeholder)
        fill_report[col] = null_count

    logger.info("Missing value handling report: %s", fill_report)
    return working