"""
Tests for outlier handling — robust z-score capping, no row drops,
group-wise computation.
"""

import pandas as pd
import numpy as np
import pytest

from src.preprocessing.outliers import handle_outliers


@pytest.fixture
def price_df():
    """DataFrame with one obvious outlier in modal_price."""
    normal_prices = [1000.0, 1050.0, 1100.0, 1080.0, 1020.0, 1060.0, 1040.0, 1010.0]
    return pd.DataFrame({
        "commodity": ["Onion"] * 9,
        "state": ["Maharashtra"] * 9,
        "modal_price": normal_prices + [99999.0],  # obvious outlier
        "min_price": [p - 100 for p in normal_prices] + [99000.0],
        "max_price": [p + 100 for p in normal_prices] + [100000.0],
    })


def test_outlier_rows_not_dropped(price_df):
    """Outliers are capped, not removed — row count must stay the same."""
    result = handle_outliers(price_df)
    assert len(result) == len(price_df)


def test_outlier_price_is_capped(price_df):
    """The extreme outlier value must be reduced after capping."""
    result = handle_outliers(price_df)
    assert result["modal_price"].max() < 99999.0


def test_normal_prices_unchanged(price_df):
    """Non-outlier prices must not be modified."""
    result = handle_outliers(price_df)
    normal_rows = result.iloc[:-1]
    original_normal = price_df.iloc[:-1]
    pd.testing.assert_series_equal(
        normal_rows["modal_price"].reset_index(drop=True),
        original_normal["modal_price"].reset_index(drop=True),
    )


def test_no_mutation_of_input(price_df):
    original_max = price_df["modal_price"].max()
    handle_outliers(price_df)
    assert price_df["modal_price"].max() == original_max