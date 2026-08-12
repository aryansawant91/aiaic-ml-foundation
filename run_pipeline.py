"""
Pipeline runner: single entry point to go from raw CSV data to a
trained, served-ready model in one command.

Usage:
    python run_pipeline.py                  # full pipeline
    python run_pipeline.py --stage ingest   # stop after ingestion
    python run_pipeline.py --stage validate # stop after validation
    python run_pipeline.py --stage preprocess
    python run_pipeline.py --stage features
    python run_pipeline.py --stage train    # full pipeline (default)

This is what you run before starting the API, and what the
REVIEW_PACKET demo video should show end to end.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level="INFO",
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("run_pipeline")

STAGES = ["ingest", "validate", "preprocess", "features", "train"]


def parse_args():
    parser = argparse.ArgumentParser(description="AIAIC ML Foundation Pipeline Runner")
    parser.add_argument(
        "--stage",
        choices=STAGES,
        default="train",
        help="Run pipeline up to and including this stage (default: train = full pipeline)",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save a JSON run report to data/versions/pipeline_report.json",
    )
    return parser.parse_args()


def run(stop_at: str, save_report: bool):
    report = {}
    start_total = time.time()

    # ------------------------------------------------------------------ #
    # STAGE 1 — INGESTION                                                  #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STAGE 1/5 — INGESTION")
    logger.info("=" * 60)
    t0 = time.time()

    from src.ingestion.pipeline import ingest_raw_dataset
    df_raw, manifest_raw = ingest_raw_dataset()

    report["ingestion"] = {
        "rows": len(df_raw),
        "columns": list(df_raw.columns),
        "source": manifest_raw["extra"].get("source_priority_used"),
        "hash": manifest_raw["content_hash"][:12],
        "duration_sec": round(time.time() - t0, 2),
    }
    logger.info(
        "Ingestion done: %d rows | hash=%s | source=%s",
        len(df_raw),
        manifest_raw["content_hash"][:12],
        manifest_raw["extra"].get("source_priority_used"),
    )

    if stop_at == "ingest":
        _finish(report, save_report, start_total)
        return

    # ------------------------------------------------------------------ #
    # STAGE 2 — VALIDATION                                                 #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STAGE 2/5 — VALIDATION")
    logger.info("=" * 60)
    t0 = time.time()

    from src.validation.pipeline import validate_dataset
    df_valid, df_quarantined, manifest_valid = validate_dataset(
        df_raw, parent_hash=manifest_raw["content_hash"]
    )

    report["validation"] = {
        "clean_rows": len(df_valid),
        "quarantined_rows": len(df_quarantined),
        "quarantine_reasons": manifest_valid["extra"].get("quarantine_reasons", {}),
        "hash": manifest_valid["content_hash"][:12],
        "duration_sec": round(time.time() - t0, 2),
    }

    if df_quarantined is not None and len(df_quarantined) > 0:
        quarantine_path = Path("data/processed/quarantined_rows.csv")
        df_quarantined.to_csv(quarantine_path, index=False)
        logger.info("Quarantined rows saved to %s", quarantine_path)

    logger.info(
        "Validation done: %d clean | %d quarantined | hash=%s",
        len(df_valid),
        len(df_quarantined),
        manifest_valid["content_hash"][:12],
    )

    if stop_at == "validate":
        _finish(report, save_report, start_total)
        return

    # ------------------------------------------------------------------ #
    # STAGE 3 — PREPROCESSING                                              #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STAGE 3/5 — PREPROCESSING")
    logger.info("=" * 60)
    t0 = time.time()

    from src.preprocessing.pipeline import run_preprocessing
    df_processed, manifest_processed = run_preprocessing(
        df_valid, parent_hash=manifest_valid["content_hash"]
    )

    processed_path = Path("data/processed/processed.csv")
    df_processed.to_csv(processed_path, index=False)

    report["preprocessing"] = {
        "rows": len(df_processed),
        "hash": manifest_processed["content_hash"][:12],
        "duration_sec": round(time.time() - t0, 2),
    }
    logger.info(
        "Preprocessing done: %d rows | hash=%s | saved to %s",
        len(df_processed),
        manifest_processed["content_hash"][:12],
        processed_path,
    )

    if stop_at == "preprocess":
        _finish(report, save_report, start_total)
        return

    # ------------------------------------------------------------------ #
    # STAGE 4 — FEATURE ENGINEERING                                        #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STAGE 4/5 — FEATURE ENGINEERING")
    logger.info("=" * 60)
    t0 = time.time()

    from src.features.pipeline import run_feature_engineering
    df_features, manifest_features = run_feature_engineering(
        df_processed, parent_hash=manifest_processed["content_hash"]
    )

    features_path = Path("data/processed/features.csv")
    df_features.to_csv(features_path, index=False)

    feature_cols = [
        c for c in df_features.columns
        if c.startswith(("modal_price_lag_", "modal_price_roll_", "month", "season", "is_weekend", "day_of_week"))
    ]
    report["features"] = {
        "rows": len(df_features),
        "feature_columns_added": feature_cols,
        "hash": manifest_features["content_hash"][:12],
        "duration_sec": round(time.time() - t0, 2),
    }
    logger.info(
        "Feature engineering done: %d rows | %d feature cols | hash=%s | saved to %s",
        len(df_features),
        len(feature_cols),
        manifest_features["content_hash"][:12],
        features_path,
    )

    if stop_at == "features":
        _finish(report, save_report, start_total)
        return

    # ------------------------------------------------------------------ #
    # STAGE 5 — TRAINING                                                   #
    # ------------------------------------------------------------------ #
    logger.info("=" * 60)
    logger.info("STAGE 5/5 — TRAINING")
    logger.info("=" * 60)
    t0 = time.time()

    from src.training.train import run_training
    result = run_training(df_features, data_hash=manifest_features["content_hash"])

    report["training"] = {
        "model_file": result["model_path"],
        "metrics": result["metrics"],
        "baseline_metrics": result["baseline_metrics"],
        "beats_baseline": result["beats_baseline"],
        "duration_sec": round(time.time() - t0, 2),
    }

    logger.info("=" * 60)
    logger.info("TRAINING RESULTS")
    logger.info("  MAE   : %.2f", result["metrics"]["mae"])
    logger.info("  RMSE  : %.2f", result["metrics"]["rmse"])
    logger.info("  MAPE  : %.2f%%", result["metrics"]["mape_pct"])
    logger.info("  R²    : %.4f", result["metrics"]["r2"])
    logger.info(
        "  Baseline MAE: %.2f  (%s)",
        result["baseline_metrics"].get("mae", float("nan")),
        "✓ model beats baseline" if result["beats_baseline"] else "✗ model does NOT beat baseline",
    )
    logger.info("  Model : %s", result["model_path"])
    logger.info("=" * 60)

    _finish(report, save_report, start_total)


def _finish(report: dict, save_report: bool, start_total: float):
    total = round(time.time() - start_total, 2)
    report["total_duration_sec"] = total
    logger.info("Pipeline complete in %.2fs", total)

    if save_report:
        report_path = Path("data/versions/pipeline_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        logger.info("Run report saved to %s", report_path)
    else:
        print("\n--- PIPELINE REPORT ---")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    args = parse_args()
    try:
        run(stop_at=args.stage, save_report=args.save_report)
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        sys.exit(1)