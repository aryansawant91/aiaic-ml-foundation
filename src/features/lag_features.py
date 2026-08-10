"""
Lag features: for each (commodity, state) time series, adds columns
for the modal_price N days prior, per settings.lag_days.

These are the single most predictive features for price forecasting —
"what was the price recently" beats almost any other signal — so
getting the grouping and sort order right here matters a lot.
"""

import logging

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

GROUP_KEYS = ["commodity", "state"]


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns like `modal_price_lag_1`, `modal_price_lag_7`, etc.

    Rows without enough history in their group (e.g. the first N days
    of a commodity/state series) will have NaN lag values — this is
    correct and expected; it is handled explicitly at training time
    (those rows are dropped, not imputed, since imputing a lag value
    would leak fabricated signal).

    Requires df to have a `date` column and be sorted by it internally
    (this function sorts a copy; it does not assume the caller sorted).
    """
    working = df.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)

    for lag in settings.lag_days:
        col_name = f"modal_price_lag_{lag}"
        working[col_name] = working.groupby(GROUP_KEYS)["modal_price"].shift(lag)
        logger.info(
            "Added %s: %d non-null of %d rows",
            col_name, working[col_name].notna().sum(), len(working),
        )

    return working