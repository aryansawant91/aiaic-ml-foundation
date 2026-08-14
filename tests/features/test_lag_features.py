"""
Tests for lag feature generation — correct shift, no leakage,
per-group computation.
"""

import pandas as pd
import pytest
from unittest.mock import patch

from src.features.lag_features import add_lag_features


@pytest.fixture
def time_series_df():
    return pd.DataFrame({
        "commodity": ["Onion"] * 5,
        "state": ["Maharashtra"] * 5,
        "date": pd.date_range("2024-01-01", periods=5, freq="D"),
        "modal_price": [1000.0, 1100.0, 1200.0, 1150.0, 1250.0],
    })


def test_lag_1_correct_values(time_series_df):
    with patch("src.features.lag_features.settings") as s:
        s.lag_days = [1]
        result = add_lag_features(time_series_df)
    # Row 1's lag_1 should equal row 0's modal_price
    assert result.iloc[1]["modal_price_lag_1"] == 1000.0
    assert result.iloc[2]["modal_price_lag_1"] == 1100.0


def test_first_row_lag_is_null(time_series_df):
    """First row of each group has no history — lag must be NaN."""
    with patch("src.features.lag_features.settings") as s:
        s.lag_days = [1]
        result = add_lag_features(time_series_df)
    assert pd.isna(result.iloc[0]["modal_price_lag_1"])


def test_separate_groups_dont_bleed():
    """Lag for group A must not use prices from group B."""
    df = pd.DataFrame({
        "commodity": ["Onion", "Onion", "Potato", "Potato"],
        "state": ["Maharashtra"] * 4,
        "date": pd.to_datetime(["2024-01-01", "2024-01-02"] * 2),
        "modal_price": [1000.0, 1100.0, 500.0, 600.0],
    })
    with patch("src.features.lag_features.settings") as s:
        s.lag_days = [1]
        result = add_lag_features(df)

    onion = result[result["commodity"] == "Onion"].sort_values("date")
    potato = result[result["commodity"] == "Potato"].sort_values("date")

    # Potato lag_1 should be 500, not 1100 (Onion's previous price)
    assert potato.iloc[1]["modal_price_lag_1"] == 500.0
    assert onion.iloc[1]["modal_price_lag_1"] == 1000.0