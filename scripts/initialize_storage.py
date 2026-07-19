"""Initialize the MachineGuard object-storage bucket and prefixes."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import S3Settings, S3StorageClient  # noqa: E402


DEFAULT_PREFIXES = [
    "raw",
    "processed",
    "reference",
    "current",
    "models",
    "drift-reports",
    "training-reports",
    "metadata",
]


def configure_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )


def main() -> int:
    """Create the bucket and standard storage prefixes."""
    load_dotenv(PROJECT_ROOT / ".env")
    configure_logging()

    try:
        settings = S3Settings.from_environment()
        storage_client = S3StorageClient(settings)

        print("\nStorage configuration:")
        print(
            json.dumps(
                settings.sanitized_summary(),
                indent=2,
                default=str,
            ),
        )

        created = storage_client.create_bucket()

        if created:
            print(
                f"\nCreated bucket: {settings.bucket_name}",
            )
        else:
            print(
                f"\nBucket already exists: {settings.bucket_name}",
            )

        initialized_prefixes = storage_client.initialize_prefixes(
            DEFAULT_PREFIXES,
        )

        storage_client.upload_json(
            data={
                "project": "MachineGuard MLOps",
                "bucket": settings.bucket_name,
                "provider": settings.provider,
                "region": settings.region,
                "initialized_at": datetime.now(
                    timezone.utc,
                ).isoformat(),
                "prefixes": DEFAULT_PREFIXES,
            },
            object_key="metadata/storage-initialization.json",
            metadata={
                "artifact-type": "storage-initialization",
            },
        )

        print("\nInitialized prefixes:")

        for prefix in initialized_prefixes:
            print(f"  - {prefix}")

        print(
            "\nStorage initialization completed successfully.",
        )
        return 0

    except Exception as error:
        logging.exception(
            "Storage initialization failed: %s",
            error,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())