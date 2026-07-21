"""Shared API client for MachineGuard."""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv(
    "API_URL",
    "https://machineguard-mlops.onrender.com",
).rstrip("/")

TIMEOUT = 30
READY_TIMEOUT = 15
MAX_RETRIES = 2


class APIError(Exception):
    """Raised when the API request fails."""


def _request_with_retry(
    method: str,
    path: str,
    timeout: int,
    retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> requests.Response:
    """Make a request with retries.

    Render free-tier services spin down after ~15 min of inactivity and
    can take 30-50s to wake up on the first hit. Without a retry, that
    first request looks like a dead API when it's actually just asleep.
    """

    last_exc: Exception | None = None

    for attempt in range(retries + 1):
        try:
            response = requests.request(
                method,
                f"{API_URL}{path}",
                timeout=timeout,
                **kwargs,
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < retries:
                time.sleep(2)
            continue

    raise APIError(str(last_exc)) from last_exc


def get_health() -> dict[str, Any]:
    """Health endpoint."""

    response = _request_with_retry("GET", "/health", timeout=READY_TIMEOUT)
    return response.json()


def get_ready() -> dict[str, Any]:
    """Ready endpoint. Retries once to survive Render cold starts."""

    response = _request_with_retry("GET", "/ready", timeout=READY_TIMEOUT)
    return response.json()


def _log_activity(kind: str, count: int, risk_level: str | None = None) -> None:
    """Record recent prediction activity for the dashboard on Home.py."""

    if "recent_activity" not in st.session_state:
        st.session_state["recent_activity"] = []

    st.session_state["recent_activity"].insert(
        0,
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "type": kind,
            "count": count,
            "risk_level": risk_level or "-",
        },
    )

    # Keep only the 10 most recent entries
    st.session_state["recent_activity"] = st.session_state["recent_activity"][:10]


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Single prediction."""

    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=payload,
            timeout=TIMEOUT,
        )
        response.raise_for_status()
        result = response.json()

        try:
            _log_activity("Single Prediction", 1, result.get("risk_level"))
        except Exception:
            pass  # dashboard logging should never break a real prediction

        return result

    except requests.RequestException as exc:
        raise APIError(str(exc)) from exc


def batch_predict(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Batch prediction."""

    results = []

    for record in records:
        prediction = predict(record)

        record["prediction"] = prediction["prediction"]
        record["failure_probability"] = prediction["failure_probability"]
        record["risk_level"] = prediction["risk_level"]

        results.append(record)

    try:
        _log_activity("Batch Prediction", len(records))
    except Exception:
        pass

    return results


def batch_predict_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction that accepts and returns a DataFrame."""

    records = dataframe.to_dict(orient="records")
    results = batch_predict(records)
    return pd.DataFrame(results)
