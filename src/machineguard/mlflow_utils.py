"""MLflow configuration utilities for MachineGuard."""

from __future__ import annotations

import os

import mlflow
from mlflow import MlflowClient


def configure_mlflow(
    default_tracking_uri: str = "sqlite:///mlflow.db",
) -> tuple[str, str]:
    """Configure MLflow tracking and registry URIs.

    Environment variables take priority over local configuration.

    Args:
        default_tracking_uri: Fallback tracking URI when the
            MLFLOW_TRACKING_URI environment variable is absent.

    Returns:
        Tuple containing the tracking URI and registry URI.
    """
    tracking_uri = os.getenv(
        "MLFLOW_TRACKING_URI",
        default_tracking_uri,
    )

    registry_uri = os.getenv(
        "MLFLOW_REGISTRY_URI",
        tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(registry_uri)

    return tracking_uri, registry_uri


def create_mlflow_client(
    default_tracking_uri: str = "sqlite:///mlflow.db",
) -> MlflowClient:
    """Create an MLflow client using configured server URIs.

    Args:
        default_tracking_uri: Fallback tracking URI when environment
            configuration is absent.

    Returns:
        Configured MLflow client.
    """
    tracking_uri, registry_uri = configure_mlflow(
        default_tracking_uri=default_tracking_uri,
    )

    return MlflowClient(
        tracking_uri=tracking_uri,
        registry_uri=registry_uri,
    )
