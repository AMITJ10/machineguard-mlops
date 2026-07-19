"""
Load the production model.

Development:
    MLflow Model Registry

Production:
    Exported Joblib pipeline
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import joblib
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel

from src.machineguard.config import load_config
from src.machineguard.mlflow_utils import configure_mlflow


APP_ENV = os.getenv("APP_ENV", "development")


@dataclass(frozen=True)
class LoadedProductionModel:

    model: object
    model_name: str
    model_alias: str
    model_version: str
    model_uri: str
    run_id: str
    tracking_uri: str
    registry_uri: str


def _get_registry_settings():

    config = load_config()

    default_tracking_uri = config["mlflow"]["tracking_uri"]

    default_registry_uri = config["mlflow"].get(
        "registry_uri",
        default_tracking_uri,
    )

    model_name = os.getenv(
        "MODEL_NAME",
        config["registry"]["model_name"],
    )

    model_alias = os.getenv(
        "MODEL_ALIAS",
        config["registry"].get(
            "champion_alias",
            "champion",
        ),
    )

    return (
        default_tracking_uri,
        model_name,
        model_alias,
        default_registry_uri,
    )


@lru_cache(maxsize=1)
def load_production_model():

    # --------------------------------------------------
    # Production (Render)
    # --------------------------------------------------

    if APP_ENV == "production":

        model_path = os.getenv(
            "MODEL_PATH",
            "artifacts/machineguard_pipeline.joblib",
        )

        pipeline = joblib.load(model_path)

        return LoadedProductionModel(
            model=pipeline,
            model_name="MachineGuardFailureClassifier",
            model_alias="production",
            model_version="joblib",
            model_uri=model_path,
            run_id="N/A",
            tracking_uri="N/A",
            registry_uri="N/A",
        )

    # --------------------------------------------------
    # Local Development (MLflow)
    # --------------------------------------------------

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

    version = client.get_model_version_by_alias(
        name=model_name,
        alias=model_alias,
    )

    model_uri = f"models:/{model_name}@{model_alias}"

    model = mlflow.pyfunc.load_model(model_uri)

    return LoadedProductionModel(
        model=model,
        model_name=model_name,
        model_alias=model_alias,
        model_version=str(version.version),
        model_uri=model_uri,
        run_id=str(version.run_id),
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )


def clear_model_cache():

    load_production_model.cache_clear()


def reload_production_model():

    clear_model_cache()

    return load_production_model()