"""Structured prediction logging utilities for MachineGuard."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTION_LOG_PATH = PROJECT_ROOT / "logs" / "predictions.jsonl"

_LOG_LOCK = Lock()


def _utc_timestamp() -> str:
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _append_record(
    record: dict[str, Any],
) -> None:
    """Append one JSON record to the prediction log.

    Args:
        record: Prediction event to write.
    """
    PREDICTION_LOG_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    serialized_record = json.dumps(
        record,
        ensure_ascii=False,
        default=str,
        separators=(",", ":"),
    )

    try:
        with _LOG_LOCK:
            with PREDICTION_LOG_PATH.open(
                mode="a",
                encoding="utf-8",
                newline="\n",
            ) as log_file:
                log_file.write(serialized_record)
                log_file.write("\n")

    except OSError:
        logger.exception(
            "Prediction event could not be written to %s.",
            PREDICTION_LOG_PATH,
        )


def log_prediction(
    *,
    features: dict[str, Any],
    prediction: int,
    probability: float,
    model_name: str,
    model_version: str,
    model_alias: str,
) -> dict[str, Any]:
    """Log a successful prediction using the legacy interface.

    This function is retained for backward compatibility with
    existing unit tests and earlier MachineGuard components.

    Args:
        features: Input features used for prediction.
        prediction: Predicted class.
        probability: Predicted failure probability.
        model_name: Registered model name.
        model_version: Registered model version.
        model_alias: Registered model alias.

    Returns:
        The record written to the JSON Lines log.
    """
    request_id = str(uuid4())

    record: dict[str, Any] = {
        "request_id": request_id,
        "prediction_id": request_id,
        "timestamp": _utc_timestamp(),
        "status": "success",
        "features": features,
        "prediction": int(prediction),
        "probability": float(probability),
        "failure_probability": float(probability),
        "model": {
            "name": model_name,
            "version": str(model_version),
            "alias": model_alias,
        },
        "model_name": model_name,
        "model_version": str(model_version),
        "model_alias": model_alias,
        **features,
    }

    _append_record(record)

    return record


def write_prediction_log(
    *,
    request_data: dict[str, Any],
    response_data: dict[str, Any] | None,
    latency_ms: float,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Append one prediction event to the JSON Lines log.

    Args:
        request_data: Validated machine input values.
        response_data: Prediction response, or ``None`` when
            prediction fails.
        latency_ms: Prediction latency in milliseconds.
        status: Either ``success`` or ``error``.
        error_message: Optional safe error description.

    Returns:
        The record written to the prediction log.

    Raises:
        ValueError: If status is not ``success`` or ``error``.
    """
    if status not in {
        "success",
        "error",
    }:
        raise ValueError("status must be either 'success' or 'error'.")

    request_id = str(uuid4())

    log_record: dict[str, Any] = {
        "request_id": request_id,
        "timestamp": _utc_timestamp(),
        "status": status,
        "latency_ms": round(
            float(latency_ms),
            3,
        ),
        "features": request_data.copy(),
        **request_data,
    }

    if response_data is not None:
        prediction_id = response_data.get(
            "prediction_id",
            request_id,
        )

        model_name = response_data.get("model_name")

        model_version = response_data.get("model_version")

        model_alias = response_data.get("model_alias")

        failure_probability = response_data.get("failure_probability")

        log_record.update(
            {
                "prediction_id": prediction_id,
                "prediction": response_data.get("prediction"),
                "failure_probability": (failure_probability),
                "probability": failure_probability,
                "risk_level": response_data.get("risk_level"),
                "threshold": response_data.get("threshold"),
                "model_name": model_name,
                "model_version": model_version,
                "model_alias": model_alias,
                "model": {
                    "name": model_name,
                    "version": model_version,
                    "alias": model_alias,
                },
            }
        )

    if error_message is not None:
        log_record["error_message"] = error_message

    _append_record(log_record)

    return log_record


def read_prediction_logs() -> list[dict[str, Any]]:
    """Read valid prediction records from the JSON Lines log.

    Returns:
        List of valid prediction records.
    """
    if not PREDICTION_LOG_PATH.exists():
        return []

    records: list[dict[str, Any]] = []

    try:
        with PREDICTION_LOG_PATH.open(
            mode="r",
            encoding="utf-8",
        ) as log_file:
            for line_number, line in enumerate(
                log_file,
                start=1,
            ):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                try:
                    record = json.loads(stripped_line)

                except json.JSONDecodeError:
                    logger.warning(
                        "Skipping invalid JSON on line %s in %s.",
                        line_number,
                        PREDICTION_LOG_PATH,
                    )

                    continue

                if isinstance(record, dict):
                    records.append(record)

    except OSError:
        logger.exception(
            "Prediction log could not be read from %s.",
            PREDICTION_LOG_PATH,
        )

        return []

    return records
