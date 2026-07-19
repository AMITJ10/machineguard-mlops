from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from src.storage import S3Settings, S3StorageClient


class MonitoringReportUploader:
    """Uploads drift monitoring reports to MinIO/S3."""

    def __init__(self) -> None:
        self.client = S3StorageClient(
            S3Settings.from_environment()
        )

    def upload_reports(
        self,
        report_directory: str | Path,
    ) -> dict[str, dict]:
        report_directory = Path(report_directory)

        timestamp = datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        destination_prefix = (
            f"drift-reports/{timestamp}"
        )

        uploaded: dict[str, dict] = {}

        for filename in [
            "data_drift_report.html",
            "data_drift_report.json",
            "data_drift_summary.json",
        ]:

            file_path = report_directory / filename

            uploaded[filename] = self.client.upload_file(
                local_path=file_path,
                object_key=f"{destination_prefix}/{filename}",
                metadata={
                    "pipeline": "machineguard",
                    "artifact": "drift-monitoring",
                },
            )

        return uploaded