"""Utilities for uploading approved ML artifacts to object storage."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.storage import S3Settings, S3StorageClient


class ArtifactUploadError(RuntimeError):
    """Raised when production artifact upload fails."""


class ModelArtifactUploader:
    """Uploads approved model artifacts and metadata to S3 or MinIO."""

    def __init__(self) -> None:
        settings = S3Settings.from_environment()
        self.client = S3StorageClient(settings)

    @staticmethod
    def _require_file(path: str | Path) -> Path:
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(
                f"Artifact does not exist: {file_path}"
            )

        if not file_path.is_file():
            raise ValueError(
                f"Artifact path is not a file: {file_path}"
            )

        return file_path

    def upload_model(
        self,
        model_path: str | Path,
        model_version: str,
    ) -> str:
        """Upload an approved serialized model."""

        path = self._require_file(model_path)

        object_key = (
            f"models/{model_version}/{path.name}"
        )

        self.client.upload_file(
            file_path=path,
            object_key=object_key,
        )

        return object_key

    def upload_report(
        self,
        report_path: str | Path,
        model_version: str,
    ) -> str:
        """Upload a training or evaluation report."""

        path = self._require_file(report_path)

        object_key = (
            f"training-reports/{model_version}/{path.name}"
        )

        self.client.upload_file(
            file_path=path,
            object_key=object_key,
        )

        return object_key

    def upload_metadata(
        self,
        metadata: dict[str, Any],
        model_version: str,
    ) -> str:
        """Upload production model metadata."""

        payload = {
            **metadata,
            "model_version": model_version,
            "uploaded_at": datetime.now(timezone.utc).isoformat(),
        }

        object_key = (
            f"metadata/models/{model_version}/metadata.json"
        )

        self.client.upload_json(
            data=payload,
            object_key=object_key,
        )

        return object_key

    def upload_manifest(
        self,
        manifest: dict[str, Any],
        model_version: str,
    ) -> str:
        """Upload a deployment manifest."""

        serialized_manifest = json.loads(
            json.dumps(manifest, default=str)
        )

        object_key = (
            f"metadata/models/{model_version}/manifest.json"
        )

        self.client.upload_json(
            data=serialized_manifest,
            object_key=object_key,
        )

        return object_key