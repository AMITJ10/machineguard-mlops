"""Shared pytest fixtures for MachineGuard tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest


class FakePyFuncModel:
    """Small fake replacement for an MLflow PyFunc model."""

    def predict(
        self,
        model_input: Any,
    ) -> np.ndarray:
        """Return deterministic class probabilities."""
        row_count = len(model_input)

        return np.tile(
            np.array([[0.90, 0.10]]),
            (row_count, 1),
        )


@pytest.fixture(autouse=True)
def mock_production_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent API tests from contacting the real MLflow server."""
    import api.main as main_module

    loaded_model = SimpleNamespace(
        model=FakePyFuncModel(),
        model_name="MachineGuardFailureClassifier",
        model_alias="champion",
        model_version="1",
        model_uri=("models:/MachineGuardFailureClassifier@champion"),
        run_id="test-run-id",
        tracking_uri="http://mock-mlflow:5000",
        registry_uri="http://mock-mlflow:5000",
    )

    monkeypatch.setattr(
        main_module,
        "load_production_model",
        lambda: loaded_model,
    )

    if hasattr(
        main_module,
        "reload_production_model",
    ):
        monkeypatch.setattr(
            main_module,
            "reload_production_model",
            lambda: loaded_model,
        )
