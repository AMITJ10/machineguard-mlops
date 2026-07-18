from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator

from api.model_loader import load_production_model
from api.prediction_service import (
    predict_failure,
    predict_failure_batch,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    ErrorResponse,
    HealthResponse,
    MachinePredictionRequest,
    MachinePredictionResponse,
    ReadinessResponse,
    RootResponse,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(
    app: FastAPI,
) -> AsyncIterator[None]:
    """Load the production model during API startup."""
    del app

    try:
        production_model = load_production_model()

        logger.info(
            "Loaded model %s version %s using alias %s.",
            production_model.model_name,
            production_model.model_version,
            production_model.model_alias,
        )
    except Exception:
        logger.exception("The production model could not be loaded.")

        raise

    yield


app = FastAPI(
    title="MachineGuard Prediction API",
    version="1.0.0",
    description=(
        "Predictive-maintenance API for estimating industrial machine-failure risk."
    ),
    lifespan=lifespan,
)


@app.get(
    "/",
    response_model=RootResponse,
    tags=["Service"],
)
def root() -> RootResponse:
    """Return API navigation information."""
    return RootResponse(
        application="MachineGuard Prediction API",
        version=app.version,
        documentation="/docs",
        health="/health",
        readiness="/ready",
        metrics="/metrics",
    )


@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Service"],
)
def health() -> HealthResponse:
    """Return the process health status."""
    return HealthResponse(
        status="healthy",
        service="machineguard-api",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


@app.get(
    "/ready",
    response_model=ReadinessResponse,
    responses={
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": ErrorResponse,
        }
    },
    tags=["Service"],
)
def readiness() -> ReadinessResponse:
    """Confirm that the champion model is available."""
    try:
        production_model = load_production_model()

        return ReadinessResponse(
            status="ready",
            model_loaded=True,
            model_name=production_model.model_name,
            model_alias=production_model.model_alias,
            model_version=production_model.model_version,
        )

    except Exception as error:
        logger.exception("Model readiness check failed.")

        raise HTTPException(
            status_code=(status.HTTP_503_SERVICE_UNAVAILABLE),
            detail={
                "error": "model_unavailable",
                "message": ("The production model is not available."),
            },
        ) from error


@app.post(
    "/predict",
    response_model=MachinePredictionResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
        }
    },
    tags=["Prediction"],
)
def predict(
    machine: MachinePredictionRequest,
) -> MachinePredictionResponse:
    """Predict failure risk for one machine."""
    try:
        return predict_failure(machine)

    except Exception as error:
        logger.exception("Single-machine prediction failed.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail={
                "error": "prediction_failed",
                "message": ("Prediction could not be completed."),
            },
        ) from error


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": ErrorResponse,
        }
    },
    tags=["Prediction"],
)
def predict_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict failure risks for multiple machines."""
    try:
        return predict_failure_batch(request)

    except Exception as error:
        logger.exception("Batch prediction failed.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail={
                "error": "batch_prediction_failed",
                "message": ("Batch prediction could not be completed."),
            },
        ) from error


Instrumentator().instrument(app).expose(
    app,
    endpoint="/metrics",
    include_in_schema=False,
)
