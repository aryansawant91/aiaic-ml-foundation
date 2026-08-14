"""
API routes for the AIAIC ML Foundation prediction service.
"""

import logging

from fastapi import APIRouter, HTTPException

from src.api.schemas import PredictionRequest, PredictionResponse, HealthResponse
from src.inference.predictor import predict_price, PredictionError, _get_model_and_metadata
from src.storage.mongo_client import mongo_is_available

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """
    Reports service health, whether a model is loaded, and whether
    Mongo is reachable. Never raises — a health check that itself
    fails is worse than useless during deployment debugging.
    """
    model_loaded = False
    model_file = None
    try:
        _, metadata = _get_model_and_metadata()
        model_loaded = True
        model_file = metadata.get("model_file")
    except Exception as e:
        logger.warning("Health check: no model loaded (%s)", e)

    return HealthResponse(
        status="ok" if model_loaded else "degraded",
        model_loaded=model_loaded,
        model_file=model_file,
        mongo_available=mongo_is_available(),
    )


@router.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Predicts modal price for a given commodity/state/date context.
    Returns 503 if no model has been trained yet, 422 if the model
    itself fails on the given input (should be rare given Pydantic
    validation, but inference-time failures are still possible).
    """
    payload = request.model_dump()

    try:
        result = predict_price(payload)
    except PredictionError as e:
        message = str(e)
        if "No trained model found" in message:
            raise HTTPException(status_code=503, detail=message)
        raise HTTPException(status_code=422, detail=message)

    return PredictionResponse(**result)