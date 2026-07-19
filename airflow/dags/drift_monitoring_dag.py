from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.sdk import dag, task


PROJECT_ROOT = Path(
    os.getenv(
        "PROJECT_ROOT",
        "/opt/airflow/machineguard",
    )
)

REPORT_DIRECTORY = PROJECT_ROOT / "monitoring" / "reports"
DRIFT_SCRIPT = PROJECT_ROOT / "scripts" / "detect_drift.py"

REFERENCE_DATA_PATH = (
    PROJECT_ROOT / "data" / "reference" / "reference.csv"
)
CURRENT_DATA_PATH = (
    PROJECT_ROOT / "data" / "current" / "current.csv"
)

SUMMARY_PATH = (
    REPORT_DIRECTORY / "data_drift_summary.json"
)


default_args = {
    "owner": "machineguard",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="machineguard_data_drift_monitoring",
    description=(
        "Detect data drift, evaluate the drift gate, "
        "and upload monitoring reports to MinIO."
    ),
    schedule="0 6 * * *",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args=default_args,
    tags=[
        "machineguard",
        "monitoring",
        "drift",
        "minio",
    ],
)
def drift_monitoring_pipeline() -> None:
    """Define the MachineGuard drift-monitoring pipeline."""

    @task
    def validate_monitoring_inputs() -> dict[str, Any]:
        """Validate all files required by the monitoring pipeline."""

        required_paths = {
            "project_root": PROJECT_ROOT,
            "drift_script": DRIFT_SCRIPT,
            "reference_data": REFERENCE_DATA_PATH,
            "current_data": CURRENT_DATA_PATH,
        }

        missing_paths = [
            str(path)
            for path in required_paths.values()
            if not path.exists()
        ]

        if missing_paths:
            raise FileNotFoundError(
                "Required monitoring files are missing: "
                + ", ".join(missing_paths)
            )

        REPORT_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        return {
            name: str(path)
            for name, path in required_paths.items()
        }

    @task
    def detect_data_drift(
        validation_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Run the Evidently drift-detection script."""

        del validation_result

        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT)

        completed_process = subprocess.run(
            [
                sys.executable,
                str(DRIFT_SCRIPT),
            ],
            cwd=str(PROJECT_ROOT),
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        print(completed_process.stdout)

        if completed_process.stderr:
            print(
                completed_process.stderr,
                file=sys.stderr,
            )

        if completed_process.returncode != 0:
            raise RuntimeError(
                "Drift detection failed with exit code "
                f"{completed_process.returncode}."
            )

        if not SUMMARY_PATH.exists():
            raise FileNotFoundError(
                f"Drift summary was not created: {SUMMARY_PATH}"
            )

        with SUMMARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as summary_file:
            summary = json.load(summary_file)

        return summary

    @task
    def evaluate_drift_gate(
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate whether dataset drift exceeded the threshold."""

        dataset_drift_detected = bool(
            summary.get(
                "dataset_drift_detected",
                False,
            )
        )

        drift_share = float(
            summary.get(
                "share_of_drifted_columns",
                0.0,
            )
        )

        drift_threshold = float(
            summary.get(
                "drift_share_threshold",
                0.3,
            )
        )

        gate_result = {
            "dataset_drift_detected": (
                dataset_drift_detected
            ),
            "share_of_drifted_columns": drift_share,
            "drift_share_threshold": drift_threshold,
            "gate_status": (
                "drift_detected"
                if dataset_drift_detected
                else "passed"
            ),
        }

        print(
            json.dumps(
                gate_result,
                indent=2,
            )
        )

        return gate_result

    @task
    def upload_monitoring_reports(
        gate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Upload generated monitoring reports to MinIO/S3."""

        from src.monitoring import (
            MonitoringReportUploader,
        )

        uploader = MonitoringReportUploader()

        upload_results = uploader.upload_reports(
            report_directory=REPORT_DIRECTORY,
        )

        result = {
            "gate_result": gate_result,
            "uploaded_reports": upload_results,
        }

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

        return result

    validation = validate_monitoring_inputs()
    drift_summary = detect_data_drift(validation)
    gate_result = evaluate_drift_gate(drift_summary)
    upload_monitoring_reports(gate_result)


drift_monitoring_pipeline()