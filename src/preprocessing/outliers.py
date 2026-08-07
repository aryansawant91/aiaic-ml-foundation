"""
Outlier handling for price columns.

Uses a robust z-score (median + MAD, not mean + std) computed within
each (commodity, state) group, since price scale varies wildly across
crops — a global threshold would flag half of one commodity and none
of another. Outliers are capped (winsorized), not dropped, to avoid
losing rows that are otherwise valid across other columns.
"""

import logging

import numpy as np
import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

PRICE_COLUMNS = ["min_price", "max_price", "modal_price"]


def _robust_z_scores(series: pd.Series) -> pd.Series:
    median = series.median()
    mad = (series - median).abs().median()
    if mad == 0:
        # Avoid div-by-zero when a group has near-constant prices.
        return pd.Series(np.zeros(len(series)), index=series.index)
    # 0.6745 scales MAD to be comparable to standard deviation under normality.
    return 0.6745 * (series - median) / mad


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each price column, computes a robust z-score within
    (commodity, state) groups and caps values beyond
    settings.outlier_z_threshold at the group's threshold bound.

    Returns a new dataframe with an added `_outlier_capped_count`
    logged (not stored in the dataframe) for transparency.
    """
    working = df.copy()
    capped_counts = {}

    for col in PRICE_COLUMNS:
        if col not in working.columns:
            continue

        z_scores = working.groupby(["commodity", "state"])[col].transform(_robust_z_scores)
        is_outlier = z_scores.abs() > settings.outlier_z_threshold
        capped_counts[col] = int(is_outlier.sum())

        if is_outlier.any():
            group_median = working.groupby(["commodity", "state"])[col].transform("median")
            group_mad = working.groupby(["commodity", "state"])[col].transform(
                lambda s: (s - s.median()).abs().median()
            )
            upper_bound = group_median + settings.outlier_z_threshold * group_mad / 0.6745
            lower_bound = group_median - settings.outlier_z_threshold * group_mad / 0.6745

            over = is_outlier & (working[col] > group_median)
            under = is_outlier & (working[col] <= group_median)
            working.loc[over, col] = upper_bound[over]
            working.loc[under, col] = lower_bound[under]

    logger.info("Outlier capping report (rows capped per column): %s", capped_counts)
    return working