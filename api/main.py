"""MachineGuard FastAPI application."""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncIterator

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DRIFT_SUMMARY_PATH = PROJECT_ROOT / "reports" / "drift" / "drift_summary.json"


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


@app.get(
    "/monitoring/drift",
    response_model=dict[str, Any],
    responses={
        status.HTTP_404_NOT_FOUND: {"description": ("No drift report is available.")},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": ("The drift summary file is invalid.")
        },
    },
    tags=["Monitoring"],
)
def get_drift_summary() -> dict[str, Any]:
    """Return the latest generated drift summary."""
    if not DRIFT_SUMMARY_PATH.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "drift_report_not_found",
                "message": (
                    "No drift report is available. Run monitoring/drift.py first."
                ),
            },
        )

    try:
        summary = json.loads(
            DRIFT_SUMMARY_PATH.read_text(
                encoding="utf-8",
            )
        )

    except json.JSONDecodeError as error:
        logger.exception("The drift summary contains invalid JSON.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail={
                "error": "invalid_drift_summary",
                "message": ("The drift summary file is invalid."),
            },
        ) from error

    except OSError as error:
        logger.exception("The drift summary could not be read.")

        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail={
                "error": "drift_summary_read_failed",
                "message": ("The drift summary could not be read."),
            },
        ) from error

    if not isinstance(summary, dict):
        raise HTTPException(
            status_code=(status.HTTP_500_INTERNAL_SERVER_ERROR),
            detail={
                "error": "invalid_drift_summary",
                "message": ("The drift summary must be a JSON object."),
            },
        )

    return summary


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
