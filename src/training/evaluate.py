"""
Evaluation metrics for the price regression model.

Kept separate from train.py so the same metric functions can be reused
for offline evaluation, inference-time monitoring, and tests.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Returns MAE, RMSE, MAPE, and R^2. MAPE is computed with a small
    epsilon guard since real mandi prices are never zero but defensive
    coding here avoids a divide-by-zero crash on bad data.
    """
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)

    mae = mean_absolute_error(y_true_arr, y_pred_arr)
    rmse = float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr)))
    r2 = r2_score(y_true_arr, y_pred_arr)

    epsilon = 1e-6
    mape = float(
        np.mean(np.abs((y_true_arr - y_pred_arr) / np.maximum(np.abs(y_true_arr), epsilon))) * 100
    )

    metrics = {
        "mae": round(float(mae), 4),
        "rmse": round(rmse, 4),
        "mape_pct": round(mape, 4),
        "r2": round(float(r2), 4),
    }
    logger.info("Evaluation metrics: %s", metrics)
    return metrics


def compute_baseline_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """
    'Naive' baseline: predict yesterday's price (modal_price_lag_1) as
    today's price. Any trained model that doesn't beat this baseline
    is not adding value — this number belongs in the eval report so
    a reviewer can judge the model honestly, not just see an R^2 in
    isolation.
    """
    if "modal_price_lag_1" not in df.columns:
        return {}
    valid = df.dropna(subset=["modal_price_lag_1", "modal_price"])
    return compute_metrics(valid["modal_price"], valid["modal_price_lag_1"].values)