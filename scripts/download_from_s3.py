"""Download a MachineGuard object from S3 or MinIO."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.storage import S3Settings, S3StorageClient  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Download an object from MachineGuard storage."
        ),
    )

    parser.add_argument(
        "--key",
        required=True,
        help=(
            "Object key, for example "
            "models/model.joblib."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Local destination path.",
    )

    return parser.parse_args()


def main() -> int:
    """Run the object download."""
    load_dotenv(PROJECT_ROOT / ".env")

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s | %(levelname)s | "
            "%(name)s | %(message)s"
        ),
    )

    arguments = parse_arguments()

    try:
        settings = S3Settings.from_environment()
        storage_client = S3StorageClient(settings)

        downloaded_path = storage_client.download_file(
            object_key=arguments.key,
            local_path=arguments.output,
        )

        print(
            f"Downloaded successfully: {downloaded_path}",
        )
        return 0

    except Exception as error:
        logging.exception("Download failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())