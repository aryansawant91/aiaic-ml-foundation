"""
Tests for dataset version registry — hashing, manifest writing, lineage.
"""

import json
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

from src.ingestion.dataset_registry import (
    _hash_dataframe,
    register_dataset_version,
    load_manifest,
)


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "commodity": ["Onion", "Potato", "Tomato"],
        "state": ["Maharashtra", "UP", "Karnataka"],
        "modal_price": [1200.0, 800.0, 600.0],
        "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
    })


def test_hash_is_deterministic(sample_df):
    """Same dataframe always produces same hash."""
    h1 = _hash_dataframe(sample_df)
    h2 = _hash_dataframe(sample_df)
    assert h1 == h2


def test_hash_changes_on_data_change(sample_df):
    """Different data produces different hash."""
    h1 = _hash_dataframe(sample_df)
    modified = sample_df.copy()
    modified.loc[0, "modal_price"] = 9999.0
    h2 = _hash_dataframe(modified)
    assert h1 != h2


def test_hash_stable_regardless_of_row_order(sample_df):
    """Hash must be the same even if rows are shuffled."""
    shuffled = sample_df.sample(frac=1, random_state=42).reset_index(drop=True)
    assert _hash_dataframe(sample_df) == _hash_dataframe(shuffled)


def test_register_writes_manifest(sample_df, tmp_path):
    """register_dataset_version writes a JSON manifest to versions_dir."""
    with patch("src.ingestion.dataset_registry.settings") as mock_settings:
        mock_settings.versions_dir = tmp_path
        with patch("src.ingestion.dataset_registry.get_mongo_collection", return_value=None):
            manifest = register_dataset_version(
                sample_df, source_name="test_source", stage="raw"
            )

    assert "content_hash" in manifest
    assert manifest["row_count"] == 3
    assert manifest["stage"] == "raw"
    assert manifest["source_name"] == "test_source"

    written_files = list(tmp_path.glob("raw_*.json"))
    assert len(written_files) == 1
    written = json.loads(written_files[0].read_text())
    assert written["content_hash"] == manifest["content_hash"]


def test_register_records_parent_hash(sample_df, tmp_path):
    """parent_hash is recorded in manifest for lineage tracing."""
    with patch("src.ingestion.dataset_registry.settings") as mock_settings:
        mock_settings.versions_dir = tmp_path
        with patch("src.ingestion.dataset_registry.get_mongo_collection", return_value=None):
            manifest = register_dataset_version(
                sample_df,
                source_name="test",
                stage="validated",
                parent_hash="abc123",
            )
    assert manifest["parent_hash"] == "abc123"