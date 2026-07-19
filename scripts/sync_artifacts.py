"""Synchronize MachineGuard artifacts with object storage."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import S3Settings, S3StorageClient  # noqa: E402


DEFAULT_SYNC_TARGETS = {
    "data/raw": "raw",
    "data/processed": "processed",
    "data/reference": "reference",
    "data/current": "current",
    "models": "models",
    "artifacts/drift": "drift-reports",
    "artifacts/reports": "training-reports",
}


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize local datasets, models and reports "
            "with object storage."
        ),
    )

    parser.add_argument(
        "--source",
        default=None,
        help=(
            "Optional single source directory. When omitted, "
            "all standard MachineGuard directories are synced."
        ),
    )

    parser.add_argument(
        "--prefix",
        default=None,
        help=(
            "Destination prefix required when --source is used."
        ),
    )

    return parser.parse_args()


def sync_target(
    storage_client: S3StorageClient,
    source_directory: Path,
    destination_prefix: str,
) -> dict[str, Any]:
    """Synchronize one local directory."""
    if not source_directory.exists():
        return {
            "source": str(source_directory),
            "prefix": destination_prefix,
            "status": "skipped",
            "reason": "Source directory does not exist.",
            "uploaded_count": 0,
            "uploads": [],
        }

    uploads = storage_client.sync_directory(
        local_directory=source_directory,
        destination_prefix=destination_prefix,
    )

    return {
        "source": str(source_directory),
        "prefix": destination_prefix,
        "status": "completed",
        "uploaded_count": len(uploads),
        "uploads": uploads,
    }


def main() -> int:
    """Synchronize configured local artifact directories."""
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    arguments = parse_arguments()

    if bool(arguments.source) != bool(arguments.prefix):
        print(
            "Error: --source and --prefix must be provided together.",
        )
        return 2

    try:
        settings = S3Settings.from_environment()
        storage_client = S3StorageClient(settings)

        if not storage_client.bucket_exists():
            storage_client.create_bucket()

        sync_results: list[dict[str, Any]] = []

        if arguments.source and arguments.prefix:
            source_path = Path(arguments.source)

            if not source_path.is_absolute():
                source_path = PROJECT_ROOT / source_path

            sync_results.append(
                sync_target(
                    storage_client=storage_client,
                    source_directory=source_path,
                    destination_prefix=arguments.prefix,
                ),
            )
        else:
            for local_directory, storage_prefix in (
                DEFAULT_SYNC_TARGETS.items()
            ):
                sync_results.append(
                    sync_target(
                        storage_client=storage_client,
                        source_directory=(
                            PROJECT_ROOT / local_directory
                        ),
                        destination_prefix=storage_prefix,
                    ),
                )

        uploaded_count = sum(
            result["uploaded_count"]
            for result in sync_results
        )

        manifest = {
            "project": "MachineGuard MLOps",
            "created_at": datetime.now(
                timezone.utc,
            ).isoformat(),
            "provider": settings.provider,
            "bucket": settings.bucket_name,
            "uploaded_count": uploaded_count,
            "sync_results": sync_results,
        }

        manifest_key = (
            "metadata/sync-manifests/"
            f"sync-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
        )

        storage_client.upload_json(
            data=manifest,
            object_key=manifest_key,
            metadata={
                "artifact-type": "sync-manifest",
            },
        )

        print(
            json.dumps(
                {
                    "uploaded_count": uploaded_count,
                    "manifest": storage_client.build_uri(
                        manifest_key,
                    ),
                    "results": sync_results,
                },
                indent=2,
                default=str,
            ),
        )

        return 0

    except Exception as error:
        logging.exception(
            "Artifact synchronization failed: %s",
            error,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())