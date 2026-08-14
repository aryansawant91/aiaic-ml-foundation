"""
Tests for missing value handling — group median fill, placeholder fill,
target column (modal_price) never imputed.
"""

import pandas as pd
import pytest

from src.preprocessing.missing_values import handle_missing_values


@pytest.fixture
def base_df():
    return pd.DataFrame({
        "commodity": ["Onion"] * 6,
        "state": ["Maharashtra"] * 6,
        "modal_price": [1000.0, 1100.0, 1200.0, 1050.0, 1150.0, 1250.0],
        "min_price": [900.0, None, 1100.0, 950.0, None, 1150.0],
        "max_price": [1100.0, 1200.0, None, 1150.0, 1250.0, None],
        "district": [None, "nashik", "nashik", None, "pune", "pune"],
        "market": ["Lasalgaon", None, "Lasalgaon", "Lasalgaon", None, "Pune"],
        "variety": [None, None, "Local", "Local", "Local", None],
    })


def test_min_price_nulls_filled(base_df):
    result = handle_missing_values(base_df)
    assert result["min_price"].isna().sum() == 0


def test_max_price_nulls_filled(base_df):
    result = handle_missing_values(base_df)
    assert result["max_price"].isna().sum() == 0


def test_district_filled_with_placeholder(base_df):
    result = handle_missing_values(base_df)
    assert result["district"].isna().sum() == 0
    assert "UNKNOWN_DISTRICT" in result["district"].values


def test_modal_price_null_rows_dropped():
    """Rows with null modal_price (target) must be dropped, not imputed."""
    df = pd.DataFrame({
        "commodity": ["Onion", "Potato"],
        "state": ["Maharashtra", "UP"],
        "modal_price": [1200.0, None],
        "min_price": [1000.0, 700.0],
        "max_price": [1400.0, 900.0],
    })
    result = handle_missing_values(df)
    assert len(result) == 1
    assert result.iloc[0]["modal_price"] == 1200.0


def test_no_mutation_of_input(base_df):
    """Input dataframe must not be mutated."""
    original_nulls = base_df["min_price"].isna().sum()
    handle_missing_values(base_df)
    assert base_df["min_price"].isna().sum() == original_nulls