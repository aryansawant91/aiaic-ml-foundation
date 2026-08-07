"""
validation/pipeline.py — validates a raw ingested dataframe against
RAW_SCHEMA, quarantining bad rows instead of silently dropping or
crashing on them. Every quarantined row is logged with a reason so
"failure handling" (a named Phase 3 deliverable) is actually inspectable.
"""

import logging
from typing import Tuple

import pandas as pd

from src.config.settings import settings
from src.validation.schema import RAW_SCHEMA, REQUIRED_COLUMNS
from src.ingestion.dataset_registry import register_dataset_version

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    pass


def validate_dataset(df: pd.DataFrame, parent_hash: str) -> Tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Returns (clean_df, quarantined_df, manifest).

    quarantined_df has the same columns as df plus a `_quarantine_reason`
    column, so nothing is silently discarded — it's just set aside.
    """
    missing_required = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_required:
        raise ValidationError(f"Dataset is missing required columns: {missing_required}")

    working = df.copy()
    working["_quarantine_reason"] = ""

    # 1. Required-field nulls
    for col in REQUIRED_COLUMNS:
        null_mask = working[col].isna() | (working[col].astype(str).str.strip() == "")
        working.loc[null_mask & (working["_quarantine_reason"] == ""), "_quarantine_reason"] = (
            f"missing_required:{col}"
        )

    # 2. Numeric columns must actually be numeric / coercible
    for spec in RAW_SCHEMA:
        if spec.dtype == "numeric" and spec.name in working.columns:
            coerced = pd.to_numeric(working[spec.name], errors="coerce")
            bad_mask = coerced.isna() & working[spec.name].notna()
            working.loc[bad_mask & (working["_quarantine_reason"] == ""), "_quarantine_reason"] = (
                f"non_numeric:{spec.name}"
            )
            working[spec.name] = coerced

    # 3. Date column must parse
    if "date" in working.columns:
        parsed_dates = pd.to_datetime(working["date"], errors="coerce", dayfirst=True)
        bad_date_mask = parsed_dates.isna()
        working.loc[bad_date_mask & (working["_quarantine_reason"] == ""), "_quarantine_reason"] = (
            "unparseable_date"
        )
        working["date"] = parsed_dates

    # 4. Sanity bounds on modal_price
    if "modal_price" in working.columns:
        out_of_range = (
            (working["modal_price"] < settings.price_min_valid) |
            (working["modal_price"] > settings.price_max_valid)
        )
        working.loc[out_of_range & (working["_quarantine_reason"] == ""), "_quarantine_reason"] = (
            "price_out_of_valid_range"
        )

    # 5. Drop columns with too many nulls entirely (not per-row quarantine)
    for col in working.columns:
        if col == "_quarantine_reason":
            continue
        null_frac = working[col].isna().mean()
        if null_frac > settings.max_null_fraction_per_column:
            logger.warning("Dropping column '%s': %.1f%% null", col, null_frac * 100)
            working = working.drop(columns=[col])

    clean_mask = working["_quarantine_reason"] == ""
    clean_df = working[clean_mask].drop(columns=["_quarantine_reason"]).reset_index(drop=True)
    quarantined_df = working[~clean_mask].reset_index(drop=True)

    if len(clean_df) < settings.min_rows_required:
        raise ValidationError(
            f"Only {len(clean_df)} clean rows survived validation "
            f"(minimum required: {settings.min_rows_required}). "
            f"{len(quarantined_df)} rows quarantined."
        )

    manifest = register_dataset_version(
        clean_df, source_name="validation_pipeline", stage="validated",
        parent_hash=parent_hash,
        extra_meta={
            "quarantined_row_count": int(len(quarantined_df)),
            "quarantine_reasons": quarantined_df["_quarantine_reason"].value_counts().to_dict()
            if len(quarantined_df) else {},
        },
    )

    logger.info(
        "Validation complete: %d clean rows, %d quarantined",
        len(clean_df), len(quarantined_df),
    )

    return clean_df, quarantined_df, manifest