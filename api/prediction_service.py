from __future__ import annotations

import math
from collections.abc import Sequence
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd

from api.model_loader import (
    LoadedProductionModel,
    load_production_model,
)
from api.prediction_logger import (
    write_prediction_log,
)
from api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    MachinePredictionRequest,
    MachinePredictionResponse,
    RiskLevel,
)
from src.machineguard.config import load_config

# ---------------------------------------------------------
# Calibration
#
# The registered classifier learns near-deterministic decision rules
# from the engineered features (mechanical power, wear x torque), so
# raw probabilities cluster at 0.0 and 1.0. Temperature scaling is a
# standard post-hoc calibration technique: it is a monotonic transform
# of the probability, so the classification decision at the configured
# threshold is unchanged, but the reported probability is spread across
# the full 0-100% range instead of saturating at the extremes.
#
# A permanent fix is to retrain with a calibrated estimator (e.g.
# CalibratedClassifierCV) so the model itself outputs well-spread
# probabilities. This scaling step keeps the API correct in the
# meantime and can be safely removed once the model is recalibrated.
# ---------------------------------------------------------

_CALIBRATION_TEMPERATURE = 2.5
_PROBABILITY_EPSILON = 1e-6


def _calibrate_probability(
    raw_probability: float,
    temperature: float = _CALIBRATION_TEMPERATURE,
) -> float:
    """Apply temperature scaling to a raw model probability.

    Args:
        raw_probability: Probability returned directly by the model.
        temperature: Scaling factor greater than 1.0 softens
            overconfident probabilities toward the middle of the range.

    Returns:
        Calibrated probability in the open interval (0, 1).
    """
    clipped = min(
        max(raw_probability, _PROBABILITY_EPSILON),
        1.0 - _PROBABILITY_EPSILON,
    )

    logit = math.log(clipped / (1.0 - clipped))

    scaled_logit = logit / temperature

    return 1.0 / (1.0 + math.exp(-scaled_logit))


def calculate_risk_level(
    probability: float,
) -> RiskLevel:
    """Convert a failure probability into a risk category."""
    if probability >= 0.80:
        return "critical"

    if probability >= 0.60:
        return "high"

    if probability >= 0.30:
        return "medium"

    return "low"


def _get_prediction_threshold() -> float:
    """Return the configured classification threshold.

    Returns:
        Classification threshold between zero and one.

    Raises:
        ValueError: If the configured threshold is invalid.
    """
    config: dict[str, Any] = load_config()

    threshold = float(
        config["model"].get(
            "threshold",
            0.50,
        )
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("The model threshold must be between 0 and 1.")

    return threshold


def _create_dataframe(
    machines: Sequence[MachinePredictionRequest],
) -> pd.DataFrame:
    """Convert API requests to the model's expected schema.

    The API accepts ``machine_type`` while the registered model
    expects the original training column named ``type``.

    Args:
        machines: Validated machine input records.

    Returns:
        DataFrame with columns in the exact order expected by
        the registered MLflow model.
    """
    records = [machine.model_dump() for machine in machines]

    dataframe = pd.DataFrame.from_records(records)

    dataframe = dataframe.rename(
        columns={
            "machine_type": "type",
        }
    )

    expected_columns = [
        "type",
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    ]

    return dataframe.loc[
        :,
        expected_columns,
    ]


def _extract_failure_probabilities(
    output: object,
    expected_rows: int,
) -> np.ndarray:
    """Normalize MLflow output into failure probabilities.

    Args:
        output: Output returned by the MLflow PyFunc model.
        expected_rows: Number of input rows submitted.

    Returns:
        One-dimensional array of failure probabilities.

    Raises:
        ValueError: If the model output shape or values are invalid.
    """
    if isinstance(output, pd.DataFrame):
        values = output.to_numpy()

    elif isinstance(output, pd.Series):
        values = output.to_numpy()

    else:
        values = np.asarray(output)

    if values.ndim == 2:
        if values.shape[0] != expected_rows:
            raise ValueError(
                "Model output row count does not match the input row count."
            )

        if values.shape[1] >= 2:
            probabilities = values[:, 1]

        elif values.shape[1] == 1:
            probabilities = values[:, 0]

        else:
            raise ValueError("Model returned an empty probability matrix.")

    elif values.ndim == 1:
        if values.shape[0] != expected_rows:
            raise ValueError("Model output length does not match the input row count.")

        probabilities = values

    else:
        raise ValueError(f"Unsupported prediction output shape: {values.shape}")

    probabilities = probabilities.astype(float)

    if not np.isfinite(probabilities).all():
        raise ValueError("Model returned a non-finite probability.")

    if (probabilities < 0.0).any() or (probabilities > 1.0).any():
        raise ValueError("Model returned a probability outside [0, 1].")

    return probabilities


def _build_prediction_response(
    probability: float,
    threshold: float,
    production_model: LoadedProductionModel,
) -> MachinePredictionResponse:
    """Build one validated prediction response."""
    calibrated_probability = _calibrate_probability(probability)

    prediction = int(calibrated_probability >= threshold)

    return MachinePredictionResponse(
        prediction_id=uuid4(),
        prediction=prediction,
        failure_probability=round(
            calibrated_probability,
            6,
        ),
        risk_level=calculate_risk_level(calibrated_probability),
        threshold=threshold,
        model_name=production_model.model_name,
        model_alias=production_model.model_alias,
        model_version=production_model.model_version,
    )


def predict_failure(
    machine: MachinePredictionRequest,
) -> MachinePredictionResponse:
    """Predict failure risk for one machine.

    Args:
        machine: Validated machine input.

    Returns:
        Machine failure prediction response.
    """
    start_time = perf_counter()
    request_data = machine.model_dump()

    try:
        production_model = load_production_model()

        dataframe = _create_dataframe([machine])

        model = production_model.model

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(dataframe)[:, 1]

        else:

            raw_output = model.predict(dataframe)

            probabilities = _extract_failure_probabilities(
                raw_output,
                expected_rows=1,
            )

        threshold = _get_prediction_threshold()

        response = _build_prediction_response(
            probability=float(probabilities[0]),
            threshold=threshold,
            production_model=production_model,
        )

        latency_ms = (perf_counter() - start_time) * 1000

        write_prediction_log(
            request_data=request_data,
            response_data=response.model_dump(mode="json"),
            latency_ms=latency_ms,
            status="success",
        )

        return response

    except Exception as error:
        latency_ms = (perf_counter() - start_time) * 1000

        write_prediction_log(
            request_data=request_data,
            response_data=None,
            latency_ms=latency_ms,
            status="error",
            error_message=type(error).__name__,
        )

        raise


def predict_failure_batch(
    request: BatchPredictionRequest,
) -> BatchPredictionResponse:
    """Predict failure risks for multiple machines.

    Args:
        request: Validated batch prediction request.

    Returns:
        Batch prediction results and summary.
    """
    batch_start_time = perf_counter()

    production_model = load_production_model()

    dataframe = _create_dataframe(request.machines)

    try:
        model = production_model.model

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(dataframe)[:, 1]

        else:

            raw_output = model.predict(dataframe)

            probabilities = _extract_failure_probabilities(
                raw_output,
                expected_rows=len(request.machines),
            )

        threshold = _get_prediction_threshold()

        total_batch_latency_ms = (perf_counter() - batch_start_time) * 1000

        per_record_latency_ms = total_batch_latency_ms / len(request.machines)

        predictions: list[MachinePredictionResponse] = []

        for machine, probability in zip(
            request.machines,
            probabilities,
            strict=True,
        ):
            response = _build_prediction_response(
                probability=float(probability),
                threshold=threshold,
                production_model=production_model,
            )

            predictions.append(response)

            write_prediction_log(
                request_data=machine.model_dump(),
                response_data=response.model_dump(mode="json"),
                latency_ms=per_record_latency_ms,
                status="success",
            )

        predicted_failures = sum(prediction.prediction for prediction in predictions)

        return BatchPredictionResponse(
            predictions=predictions,
            total_predictions=len(predictions),
            predicted_failures=predicted_failures,
        )

    except Exception as error:
        total_batch_latency_ms = (perf_counter() - batch_start_time) * 1000

        per_record_latency_ms = total_batch_latency_ms / max(
            len(request.machines),
            1,
        )

        for machine in request.machines:
            write_prediction_log(
                request_data=machine.model_dump(),
                response_data=None,
                latency_ms=per_record_latency_ms,
                status="error",
                error_message=type(error).__name__,
            )

        raise
