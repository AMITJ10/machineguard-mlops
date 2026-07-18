from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src.machineguard.config import PROJECT_ROOT, load_config
from src.machineguard.quality_gate import evaluate_quality_gate


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON file.

    Args:
        path: JSON file path.

    Returns:
        Parsed JSON dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Required file does not exist: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def get_latest_successful_run(
    client: MlflowClient,
    experiment_id: str,
):
    """Return the latest successful MLflow run.

    Args:
        client: MLflow tracking client.
        experiment_id: Experiment identifier.

    Returns:
        Latest finished MLflow run.

    Raises:
        RuntimeError: If no successful runs exist.
    """
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if not runs:
        raise RuntimeError("No successful MLflow training runs were found.")

    return runs[0]


def wait_for_model_version(
    client: MlflowClient,
    model_name: str,
    version: str,
    timeout_seconds: int = 60,
) -> None:
    """Wait until a registered model version becomes ready.

    Args:
        client: MLflow client.
        model_name: Registered-model name.
        version: Model version.
        timeout_seconds: Maximum waiting time.

    Raises:
        TimeoutError: If registration does not finish in time.
        RuntimeError: If registration fails.
    """
    deadline = time.time() + timeout_seconds

    while time.time() < deadline:
        model_version = client.get_model_version(
            name=model_name,
            version=version,
        )

        status = str(model_version.status)

        if status == "READY":
            return

        if status == "FAILED_REGISTRATION":
            raise RuntimeError("MLflow model registration failed.")

        time.sleep(1)

    raise TimeoutError("Timed out while waiting for model registration.")


def main() -> None:
    """Evaluate and promote the latest trained model."""
    config = load_config()

    tracking_uri = str(config["mlflow"]["tracking_uri"])

    experiment_name = str(config["mlflow"]["experiment_name"])

    registry_config = config["registry"]

    model_name = str(registry_config["model_name"])

    candidate_alias = str(
        registry_config.get(
            "candidate_alias",
            "candidate",
        )
    )

    champion_alias = str(
        registry_config.get(
            "champion_alias",
            "champion",
        )
    )

    validation_metrics_path = PROJECT_ROOT / "artifacts" / "validation_metrics.json"

    quality_report_path = PROJECT_ROOT / "artifacts" / "quality_gate_report.json"

    validation_metrics = load_json(validation_metrics_path)

    quality_result = evaluate_quality_gate(
        metrics=validation_metrics,
        thresholds=config["quality_gate"],
    )

    quality_report = {
        "passed": quality_result.passed,
        "checks": quality_result.checks,
        "failures": quality_result.failures,
        "metrics": validation_metrics,
        "thresholds": config["quality_gate"],
    }

    quality_report_path.write_text(
        json.dumps(
            quality_report,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("Quality-gate report:")
    print(
        json.dumps(
            quality_report,
            indent=2,
        )
    )

    if not quality_result.passed:
        raise SystemExit("Model failed the quality gate and was not registered.")

    mlflow.set_tracking_uri(tracking_uri)

    client = MlflowClient(tracking_uri=tracking_uri)

    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    latest_run = get_latest_successful_run(
        client=client,
        experiment_id=experiment.experiment_id,
    )

    run_id = latest_run.info.run_id

    source_model_uri = f"runs:/{run_id}/model"

    try:
        client.get_registered_model(model_name)
    except MlflowException:
        client.create_registered_model(
            name=model_name,
            tags={
                "application": "predictive-maintenance",
                "framework": "scikit-learn",
            },
            description=(
                "MachineGuard predictive-maintenance failure classification model."
            ),
        )

    registered_version = mlflow.register_model(
        model_uri=source_model_uri,
        name=model_name,
    )

    version = str(registered_version.version)

    wait_for_model_version(
        client=client,
        model_name=model_name,
        version=version,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="quality_gate",
        value="PASSED",
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="source_run_id",
        value=run_id,
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="validation_roc_auc",
        value=str(validation_metrics["roc_auc"]),
    )

    client.set_model_version_tag(
        name=model_name,
        version=version,
        key="validation_recall",
        value=str(validation_metrics["recall"]),
    )

    client.set_registered_model_alias(
        name=model_name,
        alias=candidate_alias,
        version=version,
    )

    try:
        client.get_model_version_by_alias(
            name=model_name,
            alias=champion_alias,
        )
    except MlflowException:
        client.set_registered_model_alias(
            name=model_name,
            alias=champion_alias,
            version=version,
        )

        champion_message = (
            "No previous champion existed. This version was assigned as champion."
        )
    else:
        champion_message = "Existing champion preserved. Candidate alias was updated."

    print("\nModel promotion completed successfully.")
    print(f"Registered model: {model_name}")
    print(f"Registered version: {version}")
    print(f"Source run ID: {run_id}")
    print(f"Candidate alias: {candidate_alias}")
    print(champion_message)


if __name__ == "__main__":
    main()
