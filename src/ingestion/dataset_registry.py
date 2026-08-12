"""
Dataset version registry.

Every time raw data is ingested, we compute a deterministic content hash
and write a version manifest (JSON, and mirrored to Mongo if available).
This lets any training run be traced back to the exact bytes of data it
was trained on, and lets us detect silent dataset drift between runs.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd

from src.config.settings import settings
from src.storage.mongo_client import get_mongo_collection


def _hash_dataframe(df: pd.DataFrame) -> str:
    """
    Fast deterministic hash using pandas internal hash_pandas_object,
    which is ~100x faster than to_csv() on large frames. We sort values
    first so the hash is stable regardless of row order.
    """
    import hashlib
    import pandas as pd

    sortable_cols = [c for c in df.columns if df[c].dtype != "object"] or list(df.columns[:1])
    df_sorted = df.sort_values(by=sortable_cols).reset_index(drop=True)

    row_hashes = pd.util.hash_pandas_object(df_sorted, index=False)
    combined = hashlib.sha256(row_hashes.values.tobytes()).hexdigest()
    return combined


def register_dataset_version(
    df: pd.DataFrame,
    source_name: str,
    stage: str,
    parent_hash: Optional[str] = None,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Record a version manifest for a dataframe at a given pipeline stage
    (e.g. "raw", "validated", "processed", "features").

    Returns the manifest dict, which callers should propagate forward
    (as parent_hash) so lineage is traceable end to end.
    """
    content_hash = _hash_dataframe(df)
    manifest = {
        "content_hash": content_hash,
        "parent_hash": parent_hash,
        "source_name": source_name,
        "stage": stage,
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "columns": list(df.columns),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "extra": extra_meta or {},
    }

    settings.versions_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = settings.versions_dir / f"{stage}_{content_hash[:12]}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Best-effort mirror to Mongo; ingestion must not fail if Mongo is down.
    collection = get_mongo_collection("dataset_versions")
    if collection is not None:
        try:
            collection.insert_one(dict(manifest))
        except Exception:
            pass  # local JSON manifest is already the source of truth

    return manifest


def load_manifest(content_hash_prefix: str, stage: str) -> Optional[Dict[str, Any]]:
    """Look up a previously written manifest by hash prefix, for replay checks."""
    matches = list(settings.versions_dir.glob(f"{stage}_{content_hash_prefix}*.json"))
    if not matches:
        return None
    return json.loads(matches[0].read_text())