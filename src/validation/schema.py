"""
schema.py — declares the expected shape of a raw ingested dataset.
This is intentionally separate from validation logic so the schema
itself is reviewable at a glance.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    required: bool
    dtype: str  # "string" | "numeric" | "date"


RAW_SCHEMA: List[ColumnSpec] = [
    ColumnSpec("state", required=True, dtype="string"),
    ColumnSpec("district", required=False, dtype="string"),
    ColumnSpec("market", required=False, dtype="string"),
    ColumnSpec("commodity", required=True, dtype="string"),
    ColumnSpec("variety", required=False, dtype="string"),
    ColumnSpec("date", required=True, dtype="date"),
    ColumnSpec("min_price", required=False, dtype="numeric"),
    ColumnSpec("max_price", required=False, dtype="numeric"),
    ColumnSpec("modal_price", required=True, dtype="numeric"),
]

REQUIRED_COLUMNS = [c.name for c in RAW_SCHEMA if c.required]