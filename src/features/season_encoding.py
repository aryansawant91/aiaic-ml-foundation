"""
Calendar / seasonality features derived from the date column.

Agricultural prices are strongly seasonal (harvest gluts, monsoon
disruption, festival demand spikes), so these are cheap, high-signal
features that don't depend on price history at all — useful even for
new commodity/state combinations with no lag data yet.
"""

import numpy as np
import pandas as pd


def _month_to_season(month: int) -> str:
    """Indian agricultural seasons, approximated by month."""
    if month in (6, 7, 8, 9):
        return "kharif"       # monsoon sowing season
    if month in (10, 11, 12, 1):
        return "rabi"         # winter sowing season
    return "zaid"             # summer/short season (Feb-May)


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - month (1-12)
      - day_of_week (0-6)
      - is_weekend
      - season (kharif / rabi / zaid)
      - month_sin, month_cos (cyclical encoding, so December and
        January are numerically close instead of 11 apart)
    """
    working = df.copy()

    working["month"] = working["date"].dt.month
    working["day_of_week"] = working["date"].dt.dayofweek
    working["is_weekend"] = working["day_of_week"].isin([5, 6]).astype(int)
    working["season"] = working["month"].apply(_month_to_season)

    working["month_sin"] = np.sin(2 * np.pi * working["month"] / 12)
    working["month_cos"] = np.cos(2 * np.pi * working["month"] / 12)

    return working