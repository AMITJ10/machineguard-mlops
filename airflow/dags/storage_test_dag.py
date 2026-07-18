"""Verify connectivity between Apache Airflow and MinIO."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from airflow.sdk import dag, task

from src.storage import S3Settings, S3StorageClient


@dag(
    dag_id="machineguard_storage_test",
    description="Verify that Airflow can access MachineGuard object storage.",
    schedule=None,
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["machineguard", "storage", "minio"],
)
def machineguard_storage_test() -> None:
    """Define tasks used to verify MinIO connectivity."""

    @task
    def inspect_storage_configuration() -> dict[str, Any]:
        """Return the non-sensitive storage configuration."""
        settings = S3Settings.from_environment()
        return settings.sanitized_summary()

    @task
    def verify_bucket() -> dict[str, Any]:
        """Verify that the configured bucket exists."""
        settings = S3Settings.from_environment()
        storage_client = S3StorageClient(settings)

        if not storage_client.bucket_exists():
            storage_client.create_bucket()

        return {
            "provider": settings.provider,
            "bucket": settings.bucket_name,
            "endpoint_url": settings.endpoint_url,
            "bucket_exists": storage_client.bucket_exists(),
        }

    @task
    def upload_connectivity_object() -> dict[str, Any]:
        """Upload a JSON object proving Airflow-to-MinIO connectivity."""
        settings = S3Settings.from_environment()
        storage_client = S3StorageClient(settings)

        object_key = "metadata/airflow-connectivity-test.json"

        payload = {
            "application": "machineguard",
            "component": "airflow",
            "storage_provider": settings.provider,
            "status": "connected",
            "message": "Airflow successfully connected to MinIO.",
        }

        # Change `data=` to `payload=` only if your upload_json method
        # uses payload as its parameter name.
        storage_client.upload_json(
            data=payload,
            object_key=object_key,
        )

        return {
            "bucket": settings.bucket_name,
            "key": object_key,
            "uri": f"s3://{settings.bucket_name}/{object_key}",
        }

    configuration = inspect_storage_configuration()
    bucket_result = verify_bucket()
    upload_result = upload_connectivity_object()

    configuration >> bucket_result >> upload_result


machineguard_storage_test()