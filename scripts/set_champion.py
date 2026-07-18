"""Assign the champion alias to a registered MLflow model."""

from __future__ import annotations

import sys

from src.machineguard.config import load_config
from src.machineguard.mlflow_utils import create_mlflow_client


def main() -> None:
    """Assign champion alias to the latest registered model version."""
    config = load_config()

    default_tracking_uri = str(config["mlflow"]["tracking_uri"])

    model_name = str(config["registry"]["model_name"])

    champion_alias = str(config["registry"]["champion_alias"])

    client = create_mlflow_client(
        default_tracking_uri=default_tracking_uri,
    )

    versions = client.search_model_versions(f"name='{model_name}'")

    if not versions:
        raise RuntimeError(
            f"No registered versions found for model "
            f"{model_name!r}. Run training first."
        )

    latest_version = max(
        versions,
        key=lambda model_version: int(model_version.version),
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=champion_alias,
        version=latest_version.version,
    )

    print(
        f"Alias {champion_alias!r} assigned to "
        f"{model_name!r} version "
        f"{latest_version.version}."
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"Failed to assign champion alias: {error}",
            file=sys.stderr,
        )
        raise
