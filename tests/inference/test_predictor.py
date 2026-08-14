"""
Tests for the inference predictor — feature row building, missing
feature handling, model load error handling.
"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.inference.predictor import _build_feature_row, predict_price, PredictionError


FEATURE_COLUMNS = [
    "state", "commodity", "grade", "min_price", "max_price",
    "month", "day_of_week", "is_weekend", "season",
    "month_sin", "month_cos",
    "modal_price_lag_1", "modal_price_lag_7", "modal_price_lag_14",
    "modal_price_roll_mean_7", "modal_price_roll_std_7",
]


def test_build_feature_row_all_present():
    payload = {col: 1.0 for col in FEATURE_COLUMNS}
    payload["state"] = "Maharashtra"
    payload["commodity"] = "Onion"
    payload["grade"] = "FAQ"
    payload["season"] = "kharif"

    df = _build_feature_row(payload, FEATURE_COLUMNS)
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) == 1


def test_build_feature_row_missing_cols_are_null():
    payload = {"commodity": "Onion", "state": "Maharashtra"}
    df = _build_feature_row(payload, FEATURE_COLUMNS)
    assert pd.isna(df.iloc[0]["modal_price_lag_1"])


def test_build_feature_row_object_cols_cast_to_category():
    payload = {col: 1.0 for col in FEATURE_COLUMNS}
    payload["state"] = "Maharashtra"
    payload["commodity"] = "Onion"
    payload["grade"] = "FAQ"
    payload["season"] = "kharif"
    df = _build_feature_row(payload, FEATURE_COLUMNS)
    assert str(df["commodity"].dtype) == "category"
    assert str(df["state"].dtype) == "category"


def test_predict_raises_if_no_model():
    with patch("src.inference.predictor._get_model_and_metadata") as mock:
        mock.side_effect = FileNotFoundError("No trained model found")
        with pytest.raises(PredictionError, match="No trained model found"):
            predict_price({"commodity": "Onion", "state": "Maharashtra"})


def test_predict_returns_expected_keys():
    mock_model = MagicMock()
    mock_model.predict.return_value = [1234.56]
    mock_metadata = {
        "feature_columns": ["commodity", "state", "grade", "month"],
        "model_file": "test_model.joblib",
        "trained_at": "20260814T000000Z",
        "data_hash": "abc123",
    }
    with patch("src.inference.predictor._get_model_and_metadata",
               return_value=(mock_model, mock_metadata)):
        result = predict_price({
            "commodity": "Onion", "state": "Maharashtra",
            "grade": "FAQ", "month": 8
        })

    assert "predicted_modal_price" in result
    assert result["predicted_modal_price"] == 1234.56
    assert "features_used" in result
    assert "features_missing" in result