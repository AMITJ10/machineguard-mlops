"""
Load the production model.

Development:
    - MLflow Model Registry

Production:
    - Bundled Joblib pipeline
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel
from src.machineguard.cloud.s3 import download_model_from_s3
from src.machineguard.config import load_config
from src.machineguard.mlflow_utils import configure_mlflow

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class LoadedProductionModel:
    """Loaded production model and its metadata."""

    model: object
    model_name: str
    model_alias: str
    model_version: str
    model_uri: str
    run_id: str
    tracking_uri: str
    registry_uri: str


def _get_registry_settings() -> tuple[str, str, str, str]:
    """Read MLflow registry configuration."""

    config: dict[str, Any] = load_config()

    default_tracking_uri = str(config["mlflow"]["tracking_uri"])

    default_registry_uri = str(
        config["mlflow"].get(
            "registry_uri",
            default_tracking_uri,
        )
    )

    model_name = os.getenv(
        "MODEL_NAME",
        str(config["registry"]["model_name"]),
    )

    model_alias = os.getenv(
        "MODEL_ALIAS",
        str(
            config["registry"].get(
                "champion_alias",
                "champion",
            )
        ),
    )

    return (
        default_tracking_uri,
        model_name,
        model_alias,
        default_registry_uri,
    )


@lru_cache(maxsize=1)
def load_production_model() -> LoadedProductionModel:
    """
    Load the production model.

    Development:
        Loads the Champion model from the MLflow Model Registry.

    Production:
        Loads the bundled Joblib model packaged inside the Docker image.
    """

    app_env = os.getenv("APP_ENV", "development").lower()

    # ==========================================================
    # Production (Render / Docker)
    # ==========================================================

    # ==========================================================
# Production (Render / Docker)
# ==========================================================

    if app_env == "production":

        model_path = download_model_from_s3()

        pipeline = joblib.load(model_path)

        return LoadedProductionModel(
            model=pipeline,
            model_name="MachineGuardFailureClassifier",
            model_alias="champion",
            model_version="joblib",
            model_uri=str(model_path),
            run_id="N/A",
            tracking_uri="N/A",
            registry_uri="N/A",
        )

    # ==========================================================
    # Development (MLflow Registry)
    # ==========================================================

    (
        default_tracking_uri,
        model_name,
        model_alias,
        default_registry_uri,
    ) = _get_registry_settings()

    tracking_uri, configured_registry_uri = configure_mlflow(
        default_tracking_uri=default_tracking_uri,
    )

    registry_uri = os.getenv(
        "MLFLOW_REGISTRY_URI",
        configured_registry_uri or default_registry_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    client = MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )

    model_version = client.get_model_version_by_alias(
        name=model_name,
        alias=model_alias,
    )

    model_uri = f"models:/{model_name}@{model_alias}"

    model: PyFuncModel = mlflow.pyfunc.load_model(model_uri)

    return LoadedProductionModel(
        model=model,
        model_name=model_name,
        model_alias=model_alias,
        model_version=str(model_version.version),
        model_uri=model_uri,
        run_id=str(model_version.run_id),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def clear_model_cache() -> None:
    """Clear the cached model."""

    load_production_model.cache_clear()


def reload_production_model() -> LoadedProductionModel:
    """Reload the production model."""

    clear_model_cache()

    return load_production_model()