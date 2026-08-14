"""
Tests for rolling features — no leakage (shift before roll),
correct window computation, per-group isolation.
"""

import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch

from src.features.rolling_features import add_rolling_features


@pytest.fixture
def rolling_df():
    prices = [1000.0, 1100.0, 1200.0, 1150.0, 1250.0, 1300.0, 1350.0, 1280.0, 1320.0, 1400.0]
    return pd.DataFrame({
        "commodity": ["Onion"] * 10,
        "state": ["Maharashtra"] * 10,
        "date": pd.date_range("2024-01-01", periods=10, freq="D"),
        "modal_price": prices,
    })


def test_rolling_mean_col_created(rolling_df):
    with patch("src.features.rolling_features.settings") as s:
        s.rolling_windows = [3]
        result = add_rolling_features(rolling_df)
    assert "modal_price_roll_mean_3" in result.columns


def test_rolling_std_col_created(rolling_df):
    with patch("src.features.rolling_features.settings") as s:
        s.rolling_windows = [3]
        result = add_rolling_features(rolling_df)
    assert "modal_price_roll_std_3" in result.columns


def test_no_leakage_rolling_mean(rolling_df):
    """Rolling mean at row T must not include price at T (shift(1) applied)."""
    with patch("src.features.rolling_features.settings") as s:
        s.rolling_windows = [3]
        result = add_rolling_features(rolling_df)

    # Row index 3 (4th row): rolling mean of rows 0,1,2 = (1000+1100+1200)/3 = 1100
    expected = (1000.0 + 1100.0 + 1200.0) / 3
    assert abs(result.iloc[3]["modal_price_roll_mean_3"] - expected) < 0.01


def test_groups_isolated(rolling_df):
    """Rolling features for group A must not include prices from group B."""
    potato_rows = rolling_df.copy()
    potato_rows["commodity"] = "Potato"
    potato_rows["modal_price"] = 500.0
    combined = pd.concat([rolling_df, potato_rows], ignore_index=True)

    with patch("src.features.rolling_features.settings") as s:
        s.rolling_windows = [3]
        result = add_rolling_features(combined)

    potato_result = result[result["commodity"] == "Potato"]
    # Potato rolling mean should be around 500, not influenced by Onion's 1000+
    valid_potato = potato_result["modal_price_roll_mean_3"].dropna()
    assert (valid_potato < 600).all()