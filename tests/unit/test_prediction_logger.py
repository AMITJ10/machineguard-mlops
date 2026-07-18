"""Tests for prediction logging."""

import json
from pathlib import Path

from api import prediction_logger


def test_log_prediction_creates_jsonl(
    tmp_path: Path,
    monkeypatch,
) -> None:
    log_path = tmp_path / "predictions.jsonl"

    monkeypatch.setattr(
        prediction_logger,
        "PREDICTION_LOG_PATH",
        log_path,
    )

    record = prediction_logger.log_prediction(
        features={
            "rotational_speed": 1_500,
            "torque": 40.0,
        },
        prediction=1,
        probability=0.82,
        model_name="MachineGuardFailureClassifier",
        model_version="1",
        model_alias="champion",
    )

    assert log_path.exists()
    assert record["prediction"] == 1

    saved_record = json.loads(
        log_path.read_text(
            encoding="utf-8",
        ).strip()
    )

    assert saved_record["failure_probability"] == 0.82
    assert saved_record["model"]["version"] == "1"
