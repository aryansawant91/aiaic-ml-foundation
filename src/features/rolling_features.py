"""
Rolling window features: mean and std of modal_price over trailing
windows (7/14/30 days by default), computed per (commodity, state)
group and strictly using only past data (shift(1) before rolling)
so no row's rolling feature leaks its own value into itself.
"""

import logging

import pandas as pd

from src.config.settings import settings

logger = logging.getLogger(__name__)

GROUP_KEYS = ["commodity", "state"]


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds columns like `modal_price_roll_mean_7`, `modal_price_roll_std_7`.

    Critical detail: we shift(1) before rolling, so the window for row
    at time T only includes prices strictly before T. Without this
    shift, the "rolling mean" would include today's own price — a
    subtle leakage bug that inflates offline metrics and quietly fails
    in production where today's price isn't known yet at prediction time.
    """
    working = df.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)

    for window in settings.rolling_windows:
        mean_col = f"modal_price_roll_mean_{window}"
        std_col = f"modal_price_roll_std_{window}"

        shifted = working.groupby(GROUP_KEYS)["modal_price"].shift(1)
        working[mean_col] = shifted.groupby(working[GROUP_KEYS[0]]).transform(
            lambda s: s.rolling(window, min_periods=max(2, window // 2)).mean()
        ) if False else (
            working.groupby(GROUP_KEYS)["modal_price"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 2)).mean())
            .reset_index(level=GROUP_KEYS, drop=True)
        )
        working[std_col] = (
            working.groupby(GROUP_KEYS)["modal_price"]
            .apply(lambda s: s.shift(1).rolling(window, min_periods=max(2, window // 2)).std())
            .reset_index(level=GROUP_KEYS, drop=True)
        )

        logger.info(
            "Added %s / %s: %d non-null of %d rows",
            mean_col, std_col, working[mean_col].notna().sum(), len(working),
        )

    return working