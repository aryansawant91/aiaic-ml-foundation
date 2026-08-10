"""
FastAPI application entry point.

Run locally with:
    uvicorn src.api.main:app --reload --port 8000

Run in Docker via docker/Dockerfile (see CMD there).
"""

import logging

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.config.settings import settings
from src.api.routes import router

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description=(
        "ML foundation service for AIAIC: crop price prediction "
        "(model foundation layer, ready for ecosystem integration)."
    ),
)

app.include_router(router)


@app.get("/")
def root():
    return {
        "service": settings.api_title,
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    """
    Catch-all so an unexpected internal error returns a clean JSON
    500 instead of leaking a raw traceback to the caller, while still
    logging the full exception server-side for debugging.
    """
    logger.exception("Unhandled exception on %s: %s", request.url.path, exc)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})