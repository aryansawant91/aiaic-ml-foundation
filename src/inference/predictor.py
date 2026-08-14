"""
Inference module: loads the latest trained model once (cached) and
exposes a single predict() function used by both the API layer and
any batch/offline scoring script.

Kept separate from the API so inference logic is testable without
spinning up FastAPI, and reusable outside the web service entirely.
"""

import logging
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd

from src.training.persistence import load_latest_model

logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = ["commodity", "state", "season", "grade"]


class PredictionError(Exception):
    pass


@lru_cache(maxsize=1)
def _get_model_and_metadata():
    """
    Cached so the model is loaded from disk exactly once per process,
    not on every request. lru_cache is safe here because the API
    process is expected to be restarted (not hot-reloaded) when a new
    model is deployed — this is documented in known_limitations.md.
    """
    model, metadata = load_latest_model()
    return model, metadata


def reload_model() -> Dict[str, Any]:
    """Clears the cache and reloads — used by a future /reload endpoint or tests."""
    _get_model_and_metadata.cache_clear()
    model, metadata = _get_model_and_metadata()
    return metadata


CATEGORICAL_COLUMNS = ["commodity", "state", "season", "grade"]


def _build_feature_row(payload: Dict[str, Any], feature_columns: List[str]) -> pd.DataFrame:
    """
    Builds a single-row dataframe matching the exact feature_columns
    the model was trained on. All object/string columns are cast to
    category dtype to match training-time dtype expectations in LightGBM.
    """
    row = {}
    missing = []
    for col in feature_columns:
        if col in payload:
            row[col] = payload[col]
        else:
            row[col] = None
            missing.append(col)

    if missing:
        logger.warning("Prediction request missing features (set to null): %s", missing)

    df = pd.DataFrame([row])

    # Cast all known categoricals
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Safety net: cast any remaining object columns to category
    # (mirrors the same safety net in train.py _prepare_features)
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype("category")

    return df

def predict_price(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main prediction entry point. `payload` should contain values for
    as many of the model's feature_columns as the caller has —
    typically commodity, state, month, season, and recent lag/rolling
    price features computed by the caller from historical data.

    Returns predicted price plus metadata about which features were
    actually used, so a client can judge prediction confidence.
    """
    try:
        model, metadata = _get_model_and_metadata()
    except FileNotFoundError as e:
        raise PredictionError(str(e)) from e

    feature_columns = metadata.get("feature_columns")
    if not feature_columns:
        raise PredictionError("Loaded model metadata has no feature_columns recorded")

    X = _build_feature_row(payload, feature_columns)

    try:
        prediction = model.predict(X)[0]
    except Exception as e:
        raise PredictionError(f"Model inference failed: {e}") from e

    missing_features = [c for c in feature_columns if payload.get(c) is None]

    return {
        "predicted_modal_price": round(float(prediction), 2),
        "model_file": metadata.get("model_file"),
        "trained_at": metadata.get("trained_at"),
        "data_hash": metadata.get("data_hash"),
        "features_used": feature_columns,
        "features_missing": missing_features,
    }