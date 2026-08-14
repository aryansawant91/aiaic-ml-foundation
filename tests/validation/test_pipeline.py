"""
Tests for validation pipeline — quarantine logic, schema enforcement,
date parsing, price bounds, clean/quarantined split.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.validation.pipeline import validate_dataset, ValidationError


@pytest.fixture
def valid_df():
    return pd.DataFrame({
        "state": ["Maharashtra", "UP", "Karnataka"] * 200,
        "commodity": ["Onion", "Potato", "Tomato"] * 200,
        "modal_price": [1200.0, 800.0, 600.0] * 200,
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"] * 200,
        "min_price": [1000.0, 700.0, 500.0] * 200,
        "max_price": [1400.0, 900.0, 700.0] * 200,
    })


def _run_validation(df):
    # Patch register_dataset_version where it's defined, not where it's used
    with patch("src.ingestion.dataset_registry.get_mongo_collection", return_value=None), \
         patch("src.ingestion.dataset_registry.settings") as mock_settings:
        import tempfile, pathlib
        tmp = pathlib.Path(tempfile.mkdtemp())
        mock_settings.versions_dir = tmp
        return validate_dataset(df, parent_hash="parent_abc")


def test_clean_data_passes_fully(valid_df):
    clean, quarantined, manifest = _run_validation(valid_df)
    assert len(clean) == len(valid_df)
    assert len(quarantined) == 0


def test_null_modal_price_quarantined(valid_df):
    df = valid_df.copy()
    df.loc[0, "modal_price"] = None
    clean, quarantined, _ = _run_validation(df)
    assert len(quarantined) >= 1
    assert "missing_required" in quarantined.iloc[0]["_quarantine_reason"]


def test_unparseable_date_quarantined(valid_df):
    df = valid_df.copy()
    df.loc[0, "date"] = "not-a-date"
    clean, quarantined, _ = _run_validation(df)
    assert any("unparseable_date" in r for r in quarantined["_quarantine_reason"])


def test_price_out_of_range_quarantined(valid_df):
    df = valid_df.copy()
    df.loc[0, "modal_price"] = 999999999.0
    clean, quarantined, _ = _run_validation(df)
    assert any("price_out_of_valid_range" in r for r in quarantined["_quarantine_reason"])


def test_missing_required_column_raises(valid_df):
    df = valid_df.drop(columns=["commodity"])
    with pytest.raises(ValidationError, match="missing required columns"):
        _run_validation(df)


def test_insufficient_clean_rows_raises():
    """If too few rows survive, ValidationError is raised."""
    df = pd.DataFrame({
        "state": ["Maharashtra"],
        "commodity": ["Onion"],
        "modal_price": [None],
        "date": ["2024-01-01"],
    })
    with pytest.raises(ValidationError):
        _run_validation(df)