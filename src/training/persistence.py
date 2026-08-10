"""
Model persistence: save/load trained models with joblib, versioned by
timestamp + short data hash so a model file's name alone tells you
when it was trained and on what data version.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib

from src.config.settings import settings

logger = logging.getLogger(__name__)


def save_model(model: Any, feature_columns: list, data_hash: str, metrics: Dict[str, float]) -> Path:
    """
    Saves the model as a .joblib file and a companion .json metadata
    file with the same stem, so `model_X.joblib` and `model_X.json`
    always travel together.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    stem = f"{settings.model_version_prefix}_{timestamp}_{data_hash[:8]}"

    model_path = settings.models_dir / f"{stem}.joblib"
    meta_path = settings.models_dir / f"{stem}.json"

    joblib.dump(model, model_path)

    metadata = {
        "model_file": model_path.name,
        "trained_at": timestamp,
        "data_hash": data_hash,
        "feature_columns": feature_columns,
        "model_type": settings.model_type,
        "metrics": metrics,
        "random_seed": settings.random_seed,
    }
    meta_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved model to %s", model_path)
    return model_path


def load_latest_model() -> Tuple[Any, Dict[str, Any]]:
    """
    Loads the most recently trained model (by filename timestamp,
    which sorts lexicographically the same as chronologically since
    we use ISO-ish UTC format). Raises if no model exists yet.
    """
    model_files = sorted(settings.models_dir.glob(f"{settings.model_version_prefix}_*.joblib"))
    if not model_files:
        raise FileNotFoundError(
            f"No trained model found in {settings.models_dir}. Run training first."
        )

    latest_model_path = model_files[-1]
    meta_path = latest_model_path.with_suffix(".json")

    model = joblib.load(latest_model_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    logger.info("Loaded model %s", latest_model_path.name)
    return model, metadata


def load_model_by_name(model_filename: str) -> Tuple[Any, Dict[str, Any]]:
    """Loads a specific model by filename, for replay/reproduction of a past run."""
    model_path = settings.models_dir / model_filename
    meta_path = model_path.with_suffix(".json")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return model, metadata