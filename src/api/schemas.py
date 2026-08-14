"""
Pydantic request/response schemas for the FastAPI service.

Kept deliberately close to the raw feature set rather than requiring
callers to pre-compute lag/rolling features themselves in most cases —
but since the model DOES need them, we accept them as optional fields
and let predictor.py handle nulls explicitly rather than the API
layer guessing.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    commodity: str = Field(..., examples=["Onion"])
    state: str = Field(..., examples=["Maharashtra"])
    month: int = Field(..., ge=1, le=12)
    day_of_week: Optional[int] = Field(default=None, ge=0, le=6)
    is_weekend: Optional[int] = Field(default=None, ge=0, le=1)
    season: Optional[str] = Field(default=None, examples=["kharif", "rabi", "zaid"])
    month_sin: Optional[float] = None
    month_cos: Optional[float] = None
    grade: Optional[str] = Field(default="FAQ", examples=["FAQ", "Local"])

    # Recent price history — caller supplies these if known; the more
    # of these that are present, the better the prediction quality.
    modal_price_lag_1: Optional[float] = Field(default=None, description="Modal price 1 day ago")
    modal_price_lag_7: Optional[float] = Field(default=None, description="Modal price 7 days ago")
    modal_price_lag_14: Optional[float] = Field(default=None, description="Modal price 14 days ago")
    modal_price_roll_mean_7: Optional[float] = None
    modal_price_roll_std_7: Optional[float] = None
    modal_price_roll_mean_14: Optional[float] = None
    modal_price_roll_std_14: Optional[float] = None
    modal_price_roll_mean_30: Optional[float] = None
    modal_price_roll_std_30: Optional[float] = None
    min_price: Optional[float] = None
    max_price: Optional[float] = None


class PredictionResponse(BaseModel):
    predicted_modal_price: float
    model_file: Optional[str]
    trained_at: Optional[str]
    data_hash: Optional[str]
    features_used: List[str]
    features_missing: List[str]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_file: Optional[str] = None
    mongo_available: bool


class ErrorResponse(BaseModel):
    detail: str