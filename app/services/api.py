"""
MachineGuard API client.

All communication with the FastAPI backend is centralized here.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 30


class APIError(Exception):
    """Raised when the backend API returns an error."""


# --------------------------------------------------------------------
# Internal Helper
# --------------------------------------------------------------------


def _request(
    method: str,
    endpoint: str,
    **kwargs: Any,
) -> Any:
    """
    Send a request to the backend.

    Raises:
        APIError
    """

    url = f"{API_URL}{endpoint}"

    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=TIMEOUT,
            **kwargs,
        )

    except requests.RequestException as error:
        raise APIError(
            f"Could not connect to backend.\n\n{error}"
        ) from error

    if not response.ok:
        try:
            detail = response.json()
        except Exception:
            detail = response.text

        raise APIError(str(detail))

    try:
        return response.json()

    except Exception as error:
        raise APIError(
            "Backend returned an invalid response."
        ) from error


# --------------------------------------------------------------------
# Health
# --------------------------------------------------------------------


def root() -> dict:
    return _request("GET", "/")


def health() -> dict:
    return _request("GET", "/health")


def ready() -> dict:
    return _request("GET", "/ready")


# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------


def predict(
    payload: dict[str, Any],
) -> dict:
    """
    Single prediction.
    """

    return _request(
        "POST",
        "/predict",
        json=payload,
    )


def predict_batch(
    dataframe: pd.DataFrame,
) -> dict:
    """
    Batch prediction.

    Converts dataframe into the backend schema.
    """

    payload = {
        "machines": dataframe.to_dict(
            orient="records",
        )
    }

    return _request(
        "POST",
        "/predict/batch",
        json=payload,
    )


# --------------------------------------------------------------------
# Monitoring
# --------------------------------------------------------------------


def drift() -> dict:
    return _request(
        "GET",
        "/monitoring/drift",
    )


# --------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------


def metrics() -> str:
    """
    Prometheus metrics.
    """

    url = f"{API_URL}/metrics"

    try:
        response = requests.get(
            url,
            timeout=TIMEOUT,
        )

        response.raise_for_status()

        return response.text

    except requests.RequestException as error:
        raise APIError(str(error)) from error


# --------------------------------------------------------------------
# API Status
# --------------------------------------------------------------------


def api_status() -> dict:
    """
    Returns backend status.
    """

    try:

        health_response = health()

        ready_response = ready()

        return {
            "online": True,
            "health": health_response,
            "ready": ready_response,
        }

    except APIError:

        return {
            "online": False,
        }