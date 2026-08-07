"""
Optional MongoDB connection helper.

Used for two things only:
  1. dataset_versions collection — mirror of local JSON manifests
  2. training_runs collection — lightweight MLflow-style run log

The service must work fully (ingestion, training, inference) with Mongo
completely absent. Every caller treats get_mongo_collection() returning
None as a normal, expected case — not an error.
"""

from functools import lru_cache
from typing import Optional

from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure

from src.config.settings import settings


@lru_cache(maxsize=1)
def _get_client() -> Optional[MongoClient]:
    try:
        client = MongoClient(
            settings.mongo_uri,
            serverSelectionTimeoutMS=settings.mongo_connect_timeout_ms,
        )
        client.admin.command("ping")  # forces connection attempt now, not lazily later
        return client
    except (ServerSelectionTimeoutError, ConnectionFailure, Exception):
        return None


def get_mongo_collection(collection_name: str):
    client = _get_client()
    if client is None:
        return None
    return client[settings.mongo_db_name][collection_name]


def mongo_is_available() -> bool:
    return _get_client() is not None