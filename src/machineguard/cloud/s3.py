"""
AWS S3 utilities for MachineGuard.

Downloads the production model from S3 if it
does not already exist locally.
"""

from __future__ import annotations

import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LOCAL_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "machineguard_pipeline.joblib"
)


def download_model_from_s3() -> Path:
    """
    Download production model from S3.

    Returns
    -------
    Path
        Local model path.
    """

    if LOCAL_MODEL_PATH.exists():
        return LOCAL_MODEL_PATH

    bucket = os.environ["S3_BUCKET"]
    key = os.environ["MODEL_KEY"]
    region = os.environ["AWS_REGION"]

    LOCAL_MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    s3 = boto3.client(
        "s3",
        region_name=region,
    )

    try:
        s3.download_file(
            bucket,
            key,
            str(LOCAL_MODEL_PATH),
        )

    except ClientError as exc:
        raise RuntimeError(
            "Failed to download model from S3."
        ) from exc

    return LOCAL_MODEL_PATH