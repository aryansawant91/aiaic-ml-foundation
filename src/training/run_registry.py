"""
Training run registry: logs every training run's config, metrics, and
resulting model file to Mongo (if available) and always to local JSON,
mirroring the dataset_registry pattern. This is the lightweight
MLflow-style tracking the learning kit references.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

from src.config.settings import settings
from src.storage.mongo_client import get_mongo_collection

logger = logging.getLogger(__name__)

RUNS_DIR = settings.project_root / "data" / "versions" / "training_runs"


def log_training_run(
    data_hash: str,
    model_path_name: str,
    metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
    feature_columns: list,
    train_rows: int,
    test_rows: int,
) -> Dict[str, Any]:
    run_record = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "data_hash": data_hash,
        "model_file": model_path_name,
        "model_type": settings.model_type,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "feature_columns": feature_columns,
        "train_rows": train_rows,
        "test_rows": test_rows,
        "random_seed": settings.random_seed,
    }

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{run_record['run_at'].replace(':', '').replace('.', '')}"
    (RUNS_DIR / f"{run_id}.json").write_text(json.dumps(run_record, indent=2))

    collection = get_mongo_collection("training_runs")
    if collection is not None:
        try:
            collection.insert_one(dict(run_record))
        except Exception:
            pass

    logger.info("Logged training run %s: %s", run_id, metrics)
    return run_record