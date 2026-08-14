"""
Tests for CSV loader — column normalization, missing required columns,
multi-file loading.
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.csv_loader import _normalize, load_fallback_dataset, CsvLoadError


@pytest.fixture
def raw_csv_df():
    """Simulates the actual Agriculture_price_dataset.csv column names."""
    return pd.DataFrame({
        "STATE": ["Maharashtra", "UP"],
        "District Name": ["nashik", "mainpuri"],
        "Market Name": ["Lasalgaon", "Bewar"],
        "Commodity": ["Onion", "Potato"],
        "Variety": ["Local", "Local"],
        "Grade": ["FAQ", "FAQ"],
        "Min_Price": [1000.0, 700.0],
        "Max_Price": [1400.0, 900.0],
        "Modal_Price": [1200.0, 800.0],
        "Price Date": ["6/6/2023", "6/6/2023"],
    })


def test_normalize_renames_all_columns(raw_csv_df):
    normalized = _normalize(raw_csv_df)
    assert "state" in normalized.columns
    assert "commodity" in normalized.columns
    assert "modal_price" in normalized.columns
    assert "date" in normalized.columns
    assert "min_price" in normalized.columns
    assert "max_price" in normalized.columns
    # Original cased columns should be gone
    assert "STATE" not in normalized.columns
    assert "Price Date" not in normalized.columns
    assert "Modal_Price" not in normalized.columns


def test_normalize_handles_url_encoded_columns():
    """Min_x0020_Price style columns from some CSV exports must normalize."""
    df = pd.DataFrame({
        "commodity": ["Wheat"],
        "date": ["2024-01-01"],
        "Min_x0020_Price": [2000.0],
        "Max_x0020_Price": [2200.0],
        "Modal_x0020_Price": [2100.0],
    })
    normalized = _normalize(df)
    assert "min_price" in normalized.columns
    assert "max_price" in normalized.columns
    assert "modal_price" in normalized.columns


def test_load_fallback_raises_if_file_missing(tmp_path):
    with patch("src.ingestion.csv_loader.settings") as mock_settings:
        mock_settings.raw_data_dir = tmp_path
        mock_settings.fallback_csv_name = "nonexistent.csv"
        with pytest.raises(CsvLoadError, match="not found"):
            load_fallback_dataset()


def test_load_fallback_raises_if_missing_required_columns(tmp_path):
    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col1,col2\n1,2\n3,4\n")
    with patch("src.ingestion.csv_loader.settings") as mock_settings:
        mock_settings.raw_data_dir = tmp_path
        mock_settings.fallback_csv_name = "bad.csv"
        with pytest.raises(CsvLoadError, match="missing required columns"):
            load_fallback_dataset()