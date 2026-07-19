from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow.providers.standard.operators.trigger_dagrun import (
    TriggerDagRunOperator,
)
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

RETRAINING_DAG_ID = "machineguard_retraining_pipeline"


default_args = {
    "owner": "machineguard",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="machineguard_data_drift_monitoring",
    description=(
        "Detect data drift, upload reports to MinIO, "
        "and trigger retraining when drift is detected."
    ),
    schedule="0 6 * * *",
    start_date=datetime(2026, 7, 1),
    catchup=False,
    default_args=default_args,
    render_template_as_native_obj=True,
    tags=[
        "machineguard",
        "monitoring",
        "drift",
        "minio",
        "retraining",
    ],
)
def drift_monitoring_pipeline() -> None:
    """Define the MachineGuard data-drift monitoring pipeline."""

    @task
    def validate_monitoring_inputs() -> dict[str, str]:
        """Validate files required by the drift-monitoring pipeline."""

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

        validation_result = {
            name: str(path)
            for name, path in required_paths.items()
        }

        print(
            json.dumps(
                validation_result,
                indent=2,
            )
        )

        return validation_result

    @task
    def detect_data_drift(
        validation_result: dict[str, str],
    ) -> dict[str, Any]:
        """Run the Evidently drift-detection script."""

        print(
            "Monitoring input validation completed:",
            json.dumps(
                validation_result,
                indent=2,
            ),
        )

        environment = os.environ.copy()

        existing_pythonpath = environment.get(
            "PYTHONPATH",
            "",
        )

        environment["PYTHONPATH"] = os.pathsep.join(
            path
            for path in [
                str(PROJECT_ROOT),
                existing_pythonpath,
            ]
            if path
        )

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

        if completed_process.stdout:
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
                "Drift summary was not generated at: "
                f"{SUMMARY_PATH}"
            )

        with SUMMARY_PATH.open(
            "r",
            encoding="utf-8",
        ) as summary_file:
            summary: dict[str, Any] = json.load(
                summary_file
            )

        print(
            json.dumps(
                summary,
                indent=2,
            )
        )

        return summary

    @task
    def evaluate_drift_gate(
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate whether detected drift requires retraining."""

        dataset_drift_detected = bool(
            summary.get(
                "dataset_drift_detected",
                False,
            )
        )

        share_of_drifted_columns = float(
            summary.get(
                "share_of_drifted_columns",
                0.0,
            )
        )

        drift_share_threshold = float(
            summary.get(
                "drift_share_threshold",
                0.3,
            )
        )

        gate_result: dict[str, Any] = {
            "dataset_drift_detected": (
                dataset_drift_detected
            ),
            "share_of_drifted_columns": (
                share_of_drifted_columns
            ),
            "drift_share_threshold": (
                drift_share_threshold
            ),
            "number_of_columns": int(
                summary.get(
                    "number_of_columns",
                    0,
                )
            ),
            "number_of_drifted_columns": int(
                summary.get(
                    "number_of_drifted_columns",
                    0,
                )
            ),
            "drifted_columns": summary.get(
                "drifted_columns",
                [],
            ),
            "generated_at": summary.get(
                "generated_at",
            ),
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
        """Upload generated drift reports to MinIO or Amazon S3."""

        from src.monitoring import (
            MonitoringReportUploader,
        )

        uploader = MonitoringReportUploader()

        upload_results = uploader.upload_reports(
            report_directory=REPORT_DIRECTORY,
        )

        result: dict[str, Any] = {
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

    @task.branch
    def choose_retraining_path(
        gate_result: dict[str, Any],
    ) -> str:
        """
        Decide whether to trigger model retraining.
        """

        if gate_result["dataset_drift_detected"]:
            print("Dataset drift detected.")
            print("Triggering retraining pipeline.")

            return "trigger_retraining"

        print("No dataset drift detected.")
        print("Skipping retraining.")

        return "skip_retraining"

    @task(task_id="skip_retraining")
    def skip_retraining(
        gate_result: dict[str, Any],
    ) -> dict[str, Any]:
        """Record that retraining was not required."""

        result: dict[str, Any] = {
            "retraining_triggered": False,
            "reason": "Dataset drift was not detected.",
            "gate_result": gate_result,
        }

        print(
            json.dumps(
                result,
                indent=2,
                default=str,
            )
        )

        return result

    validation_result = validate_monitoring_inputs()

    drift_summary = detect_data_drift(
        validation_result=validation_result,
    )

    gate_result = evaluate_drift_gate(
        summary=drift_summary,
    )

    upload_result = upload_monitoring_reports(
        gate_result=gate_result,
    )

    branch_result = choose_retraining_path(
        gate_result=gate_result,
    )

    trigger_retraining = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id=RETRAINING_DAG_ID,
        conf={
            "trigger_source": (
                "machineguard_data_drift_monitoring"
            ),
            "dataset_drift_detected": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['dataset_drift_detected'] }}"
            ),
            "share_of_drifted_columns": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['share_of_drifted_columns'] }}"
            ),
            "drift_share_threshold": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['drift_share_threshold'] }}"
            ),
            "number_of_drifted_columns": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['number_of_drifted_columns'] }}"
            ),
            "drifted_columns": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['drifted_columns'] }}"
            ),
            "monitoring_generated_at": (
                "{{ ti.xcom_pull(task_ids='evaluate_drift_gate')['generated_at'] }}"
            ),
            "monitoring_dag_run_id": "{{ run_id }}",
        },
        wait_for_completion=False,
        reset_dag_run=False,
    )

    skipped_retraining = skip_retraining(
        gate_result=gate_result,
    )

    # Ensure uploads finish before making the branching decision.
    upload_result >> branch_result

    branch_result >> [
        trigger_retraining,
        skipped_retraining,
    ]


drift_monitoring_pipeline()