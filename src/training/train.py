"""
Training pipeline: time-based split -> feature/target prep -> model
fit -> evaluation (vs. naive baseline) -> persistence -> run logging.

Train/test split is by DATE, not random shuffling — random splits on
time series data leak future information into the training set (the
model would see "future" prices for a market and trivially ace a
random test split). This is a common ML-engineering mistake and
deliberately avoided here.
"""

import logging
from typing import Tuple

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

from src.config.settings import settings
from src.training.evaluate import compute_metrics, compute_baseline_metrics
from src.training.persistence import save_model
from src.training.run_registry import log_training_run

logger = logging.getLogger(__name__)

CATEGORICAL_COLUMNS = ["commodity", "state", "season"]
DROP_COLUMNS = ["date", "modal_price", "district", "market", "variety", "_source_file"]


def _prepare_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    Encodes categoricals (pandas category dtype, which both LightGBM
    and XGBoost can consume natively) and returns (X, y, feature_columns).
    """
    working = df.copy()

    for col in CATEGORICAL_COLUMNS:
        if col in working.columns:
            working[col] = working[col].astype("category")

    feature_columns = [
        c for c in working.columns
        if c not in DROP_COLUMNS and c != "modal_price"
    ]

    X = working[feature_columns]
    y = working["modal_price"]
    return X, y, feature_columns


def _time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Holds out the last settings.test_split_days of the overall date
    range as the test set. Applied globally (not per-group) so the
    test set represents a genuine forward-in-time forecast scenario
    for every commodity/state at once.
    """
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=settings.test_split_days)

    train_df = df[df["date"] <= cutoff].reset_index(drop=True)
    test_df = df[df["date"] > cutoff].reset_index(drop=True)

    logger.info(
        "Time-based split: cutoff=%s, train_rows=%d, test_rows=%d",
        cutoff.date(), len(train_df), len(test_df),
    )
    return train_df, test_df


def _fit_model(X_train: pd.DataFrame, y_train: pd.Series):
    if settings.model_type == "lightgbm":
        model = lgb.LGBMRegressor(
            random_state=settings.random_seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
        )
        model.fit(X_train, y_train, categorical_feature=CATEGORICAL_COLUMNS)
    elif settings.model_type == "xgboost":
        # XGBoost needs categoricals as pandas category dtype + enable_categorical=True
        model = xgb.XGBRegressor(
            random_state=settings.random_seed,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            enable_categorical=True,
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unsupported model_type: {settings.model_type}")

    return model


def run_training(df: pd.DataFrame, data_hash: str) -> dict:
    """
    Full training entry point. Returns a summary dict with metrics,
    model path, and run record — this is what the training script
    or a future /train API endpoint would call.
    """
    train_df, test_df = _time_based_split(df)

    if len(train_df) < 100 or len(test_df) < 20:
        raise RuntimeError(
            f"Insufficient data for a reliable split "
            f"(train={len(train_df)}, test={len(test_df)}). "
            f"Widen the dataset or reduce test_split_days."
        )

    X_train, y_train, feature_columns = _prepare_features(train_df)
    X_test, y_test, _ = _prepare_features(test_df)

    model = _fit_model(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = compute_metrics(y_test, predictions)
    baseline_metrics = compute_baseline_metrics(test_df)

    model_path = save_model(model, feature_columns, data_hash, metrics)

    run_record = log_training_run(
        data_hash=data_hash,
        model_path_name=model_path.name,
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        feature_columns=feature_columns,
        train_rows=len(train_df),
        test_rows=len(test_df),
    )

    beats_baseline = (
        baseline_metrics.get("mae") is not None
        and metrics["mae"] < baseline_metrics["mae"]
    )
    logger.info(
        "Training complete. Model MAE=%.2f vs baseline MAE=%.2f (%s)",
        metrics["mae"], baseline_metrics.get("mae", float("nan")),
        "beats baseline" if beats_baseline else "DOES NOT beat baseline",
    )

    return {
        "model_path": str(model_path),
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "beats_baseline": beats_baseline,
        "run_record": run_record,
    }