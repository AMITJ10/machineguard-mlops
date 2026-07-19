"""MachineGuard model retraining and promotion DAG."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from airflow.sdk import dag, task
from airflow.exceptions import AirflowException

from src.storage.artifact_uploader import ModelArtifactUploader


PROJECT_ROOT = Path("/opt/airflow/machineguard")
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODEL_DIR = ARTIFACTS_DIR / "models"
REPORT_DIR = ARTIFACTS_DIR / "reports"

MODEL_PATH = MODEL_DIR / "machineguard_model.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
VALIDATION_REPORT_PATH = REPORT_DIR / "validation_report.json"

MINIMUM_F1_SCORE = 0.80
MINIMUM_RECALL = 0.80


def run_command(command: list[str]) -> None:
    """Run a project command and fail the task on error."""

    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    print(result.stdout)

    if result.stderr:
        print(result.stderr)

    if result.returncode != 0:
        raise AirflowException(
            f"Command failed with exit code "
            f"{result.returncode}: {' '.join(command)}"
        )


@dag(
    dag_id="machineguard_retraining_pipeline",
    description=(
        "Validate data, train, evaluate, approve and upload "
        "MachineGuard model artifacts."
    ),
    schedule="0 2 * * 0",
    start_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
    catchup=False,
    max_active_runs=1,
    default_args={
        "owner": "machineguard",
        "retries": 2,
        "retry_delay": timedelta(minutes=2),
    },
    tags=["machineguard", "mlops", "retraining", "minio"],
)
def machineguard_retraining_pipeline():

    @task
    def validate_dataset() -> str:
        """Run dataset validation."""

        run_command(
            [
                "python",
                "scripts/validate_data.py",
            ]
        )

        return str(VALIDATION_REPORT_PATH)

    @task
    def train_model(
        validation_report: str,
    ) -> dict[str, str]:
        """Train the model after successful validation."""

        print(f"Validation report: {validation_report}")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)

        run_command(
            [
                "python",
                "scripts/train.py",
            ]
        )

        if not MODEL_PATH.exists():
            raise AirflowException(
                f"Training completed but model was not found: "
                f"{MODEL_PATH}"
            )

        return {
            "model_path": str(MODEL_PATH),
            "metrics_path": str(METRICS_PATH),
        }

    @task
    def evaluate_model(
        training_output: dict[str, str],
    ) -> dict[str, Any]:
        """Read evaluation metrics generated during training."""

        metrics_path = Path(training_output["metrics_path"])

        if not metrics_path.exists():
            raise AirflowException(
                f"Metrics file not found: {metrics_path}"
            )

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        print(
            json.dumps(
                metrics,
                indent=2,
            )
        )

        return {
            **training_output,
            "metrics": metrics,
        }

    @task
    def quality_gate(
        evaluation_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Reject models that do not meet quality requirements."""

        metrics = evaluation_output["metrics"]

        f1_score = float(metrics.get("f1_score", 0.0))
        recall = float(metrics.get("recall", 0.0))

        failures: list[str] = []

        if f1_score < MINIMUM_F1_SCORE:
            failures.append(
                f"F1 score {f1_score:.4f} is below "
                f"{MINIMUM_F1_SCORE:.4f}"
            )

        if recall < MINIMUM_RECALL:
            failures.append(
                f"Recall {recall:.4f} is below "
                f"{MINIMUM_RECALL:.4f}"
            )

        if failures:
            raise AirflowException(
                "Model quality gate failed: "
                + "; ".join(failures)
            )

        version = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        return {
            **evaluation_output,
            "model_version": version,
            "quality_gate": "passed",
        }

    @task
    def upload_approved_artifacts(
        approved_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Upload approved artifacts to MinIO or S3."""

        uploader = ModelArtifactUploader()

        model_version = approved_output["model_version"]
        model_path = approved_output["model_path"]
        metrics_path = approved_output["metrics_path"]

        model_key = uploader.upload_model(
            model_path=model_path,
            model_version=model_version,
        )

        metrics_key = uploader.upload_report(
            report_path=metrics_path,
            model_version=model_version,
        )

        metadata_key = uploader.upload_metadata(
            metadata={
                "quality_gate": "passed",
                "metrics": approved_output["metrics"],
                "model_object_key": model_key,
                "metrics_object_key": metrics_key,
                "pipeline": (
                    "machineguard_retraining_pipeline"
                ),
            },
            model_version=model_version,
        )

        return {
            "model_version": model_version,
            "model_object_key": model_key,
            "metrics_object_key": metrics_key,
            "metadata_object_key": metadata_key,
            "status": "uploaded",
        }

    @task
    def publish_manifest(
        upload_output: dict[str, Any],
    ) -> dict[str, Any]:
        """Publish an immutable model release manifest."""

        uploader = ModelArtifactUploader()

        manifest_key = uploader.upload_manifest(
            manifest={
                **upload_output,
                "approved_for_deployment": True,
            },
            model_version=upload_output["model_version"],
        )

        result = {
            **upload_output,
            "manifest_object_key": manifest_key,
            "status": "production_candidate",
        }

        print(json.dumps(result, indent=2))

        return result

    validation = validate_dataset()
    training = train_model(validation)
    evaluation = evaluate_model(training)
    approved = quality_gate(evaluation)
    uploaded = upload_approved_artifacts(approved)
    publish_manifest(uploaded)


machineguard_retraining_pipeline()