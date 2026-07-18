"""Load the production model from the MLflow Model Registry."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.pyfunc import PyFuncModel

from src.machineguard.config import load_config
from src.machineguard.mlflow_utils import configure_mlflow


@dataclass(frozen=True)
class LoadedProductionModel:
    """Loaded production model and its MLflow registry metadata."""

    model: PyFuncModel
    model_name: str
    model_alias: str
    model_version: str
    model_uri: str
    run_id: str
    tracking_uri: str
    registry_uri: str


def _get_registry_settings() -> tuple[str, str, str, str]:
    """Return configured MLflow and model registry settings.

    Environment variables take priority over values from config.yaml.

    Returns:
        Tuple containing:

        - Default tracking URI from config
        - Registered model name
        - Production model alias
        - Default registry URI

    Raises:
        KeyError: If required configuration values are missing.
    """
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
    """Load and cache the active production model from MLflow.

    The model is downloaded once for each API worker process.
    Subsequent requests reuse the cached model.

    Returns:
        Loaded model and its registry metadata.

    Raises:
        MlflowException: If the registered model, alias, version,
            or artifacts cannot be loaded.
    """
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

    model = mlflow.pyfunc.load_model(model_uri)

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
    """Clear the cached production model."""
    load_production_model.cache_clear()


def reload_production_model() -> LoadedProductionModel:
    """Clear the cache and reload the active production model."""
    clear_model_cache()

    return load_production_model()
