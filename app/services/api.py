"""Shared API client for MachineGuard AI."""

from __future__ import annotations

import os
from typing import Any, Union

import pandas as pd
import requests

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 30
HEALTH_TIMEOUT = 10


class APIError(Exception):
    """Raised when the API request fails."""


def get_health() -> dict[str, Any]:
    """Health endpoint check."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=HEALTH_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


def get_ready() -> dict[str, Any]:
    """Ready endpoint check."""
    try:
        response = requests.get(
            f"{API_URL}/ready",
            timeout=HEALTH_TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Single prediction request."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


def batch_predict(
    data: Union[list[dict[str, Any]], pd.DataFrame],
) -> Union[list[dict[str, Any]], pd.DataFrame]:
    """Batch prediction supporting both list of dicts and pandas DataFrame."""
    if isinstance(data, pd.DataFrame):
        results = []
        for _, row in data.iterrows():
            payload = row.to_dict()
            pred = predict(payload)
            payload["prediction"] = pred.get("prediction")
            payload["failure_probability"] = pred.get("failure_probability")
            payload["risk_level"] = pred.get("risk_level")
            results.append(payload)
        return pd.DataFrame(results)

    elif isinstance(data, list):
        results = []
        for record in data:
            record_copy = record.copy()
            pred = predict(record_copy)
            record_copy["prediction"] = pred.get("prediction")
            record_copy["failure_probability"] = pred.get("failure_probability")
            record_copy["risk_level"] = pred.get("risk_level")
            results.append(record_copy)
        return results

    else:
        raise ValueError("Data must be a list of dicts or a pandas DataFrame")