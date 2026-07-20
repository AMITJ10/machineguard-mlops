"""Shared API client for MachineGuard."""

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
    """Raised when the API request fails."""


def get_health() -> dict[str, Any]:
    """Health endpoint."""

    response = requests.get(
        f"{API_URL}/health",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def get_ready() -> dict[str, Any]:
    """Ready endpoint."""

    response = requests.get(
        f"{API_URL}/ready",
        timeout=5,
    )

    response.raise_for_status()

    return response.json()


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Single prediction."""

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
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Batch prediction."""

    results = []

    for record in records:

        prediction = predict(record)

        record["prediction"] = prediction["prediction"]
        record["failure_probability"] = prediction[
            "failure_probability"
        ]
        record["risk_level"] = prediction[
            "risk_level"
        ]

        results.append(record)

    return results

    for _, row in dataframe.iterrows():

        payload = row.to_dict()

        prediction = predict(payload)

        payload["prediction"] = prediction["prediction"]
        payload["failure_probability"] = prediction[
            "failure_probability"
        ]
        payload["risk_level"] = prediction[
            "risk_level"
        ]

        results.append(payload)

    return pd.DataFrame(results)