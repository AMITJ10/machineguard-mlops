"""Unit tests for the S3-compatible storage client."""

from __future__ import annotations

import json
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from src.storage.config import S3Settings
from src.storage.exceptions import ObjectNotFoundError
from src.storage.s3_client import (
    S3StorageClient,
    calculate_sha256,
    normalize_object_key,
)


TEST_BUCKET_NAME = "machineguard-test-artifacts"
TEST_REGION = "ap-south-1"


def create_settings() -> S3Settings:
    """Create AWS-style settings for Moto tests."""
    return S3Settings(
        bucket_name=TEST_BUCKET_NAME,
        region=TEST_REGION,
        endpoint_url=None,
        access_key_id="testing",
        secret_access_key="testing",
        session_token=None,
        use_ssl=True,
        verify_ssl=True,
        addressing_style="auto",
        provider="aws",
    )


def create_mock_client() -> S3StorageClient:
    """Create a storage client backed by Moto."""
    boto3_client = boto3.client(
        "s3",
        region_name=TEST_REGION,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )

    return S3StorageClient(
        settings=create_settings(),
        boto3_client=boto3_client,
    )


def test_normalize_object_key() -> None:
    """Object keys should use normalized forward slashes."""
    assert normalize_object_key(
        r"\models\\production\model.joblib",
    ) == "models/production/model.joblib"


def test_empty_object_key_is_rejected() -> None:
    """Empty keys should not be accepted."""
    with pytest.raises(
        ValueError,
        match="Object key cannot be empty",
    ):
        normalize_object_key("///")


def test_calculate_sha256(tmp_path: Path) -> None:
    """File checksums should be stable."""
    file_path = tmp_path / "sample.txt"
    file_path.write_text("machineguard", encoding="utf-8")

    checksum = calculate_sha256(file_path)

    assert len(checksum) == 64
    assert checksum == calculate_sha256(file_path)


@mock_aws
def test_create_bucket() -> None:
    """The client should create a missing bucket."""
    storage_client = create_mock_client()

    assert storage_client.bucket_exists() is False
    assert storage_client.create_bucket() is True
    assert storage_client.bucket_exists() is True
    assert storage_client.create_bucket() is False


@mock_aws
def test_initialize_prefixes() -> None:
    """Logical storage prefixes should be created."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    initialized = storage_client.initialize_prefixes(
        ["raw", "models"],
    )

    assert initialized == ["raw/", "models/"]
    assert storage_client.object_exists("raw/") is True
    assert storage_client.object_exists("models/") is True


@mock_aws
def test_upload_and_download_file(tmp_path: Path) -> None:
    """A local file should survive an upload and download cycle."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    source_path = tmp_path / "source.csv"
    source_path.write_text(
        "temperature,vibration\n80,0.42\n",
        encoding="utf-8",
    )

    upload_result = storage_client.upload_file(
        local_path=source_path,
        object_key="raw/source.csv",
        metadata={
            "dataset-type": "raw",
        },
    )

    assert upload_result["key"] == "raw/source.csv"
    assert upload_result["uri"] == (
        f"s3://{TEST_BUCKET_NAME}/raw/source.csv"
    )
    assert storage_client.object_exists("raw/source.csv") is True

    destination_path = tmp_path / "downloads" / "source.csv"

    storage_client.download_file(
        object_key="raw/source.csv",
        local_path=destination_path,
    )

    assert destination_path.read_text(
        encoding="utf-8",
    ) == source_path.read_text(encoding="utf-8")


@mock_aws
def test_upload_and_read_json() -> None:
    """JSON documents should be serialized and deserialized."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    payload = {
        "model": "random_forest",
        "accuracy": 0.94,
    }

    storage_client.upload_json(
        data=payload,
        object_key="metadata/model.json",
    )

    downloaded_payload = storage_client.read_json(
        "metadata/model.json",
    )

    assert downloaded_payload == payload


@mock_aws
def test_get_object_metadata(tmp_path: Path) -> None:
    """Uploaded custom metadata should be available."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    source_path = tmp_path / "model.joblib"
    source_path.write_bytes(b"example-model")

    storage_client.upload_file(
        local_path=source_path,
        object_key="models/model.joblib",
        metadata={
            "model-version": "1",
        },
    )

    metadata = storage_client.get_object_metadata(
        "models/model.joblib",
    )

    assert metadata["key"] == "models/model.joblib"
    assert metadata["metadata"]["model-version"] == "1"
    assert "sha256" in metadata["metadata"]


@mock_aws
def test_list_objects_by_prefix(tmp_path: Path) -> None:
    """Object listing should support prefix filtering."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    first_file = tmp_path / "first.txt"
    second_file = tmp_path / "second.txt"

    first_file.write_text("first", encoding="utf-8")
    second_file.write_text("second", encoding="utf-8")

    storage_client.upload_file(
        first_file,
        "raw/first.txt",
    )

    storage_client.upload_file(
        second_file,
        "models/second.txt",
    )

    raw_objects = storage_client.list_objects(
        prefix="raw/",
    )

    assert len(raw_objects) == 1
    assert raw_objects[0]["key"] == "raw/first.txt"


@mock_aws
def test_delete_object(tmp_path: Path) -> None:
    """Individual objects should be deletable."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    file_path = tmp_path / "temporary.txt"
    file_path.write_text("temporary", encoding="utf-8")

    storage_client.upload_file(
        file_path,
        "metadata/temporary.txt",
    )

    assert storage_client.object_exists(
        "metadata/temporary.txt",
    )

    storage_client.delete_object(
        "metadata/temporary.txt",
    )

    assert not storage_client.object_exists(
        "metadata/temporary.txt",
    )


@mock_aws
def test_delete_prefix(tmp_path: Path) -> None:
    """All objects under a prefix should be deleted."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    for index in range(3):
        file_path = tmp_path / f"report-{index}.json"
        file_path.write_text(
            json.dumps({"index": index}),
            encoding="utf-8",
        )

        storage_client.upload_file(
            file_path,
            f"drift-reports/report-{index}.json",
        )

    deleted_count = storage_client.delete_prefix(
        "drift-reports/",
    )

    assert deleted_count == 3
    assert storage_client.list_objects(
        "drift-reports/",
    ) == []


@mock_aws
def test_sync_directory(tmp_path: Path) -> None:
    """Directory synchronization should retain relative paths."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    source_directory = tmp_path / "reports"
    nested_directory = source_directory / "daily"

    nested_directory.mkdir(parents=True)

    (source_directory / "summary.json").write_text(
        "{}",
        encoding="utf-8",
    )

    (nested_directory / "drift.json").write_text(
        "{}",
        encoding="utf-8",
    )

    uploads = storage_client.sync_directory(
        local_directory=source_directory,
        destination_prefix="training-reports",
    )

    uploaded_keys = {
        upload["key"]
        for upload in uploads
    }

    assert uploaded_keys == {
        "training-reports/summary.json",
        "training-reports/daily/drift.json",
    }


@mock_aws
def test_download_missing_object_raises_error(
    tmp_path: Path,
) -> None:
    """Missing downloads should raise a domain-specific error."""
    storage_client = create_mock_client()
    storage_client.create_bucket()

    with pytest.raises(ObjectNotFoundError):
        storage_client.download_file(
            object_key="missing/file.csv",
            local_path=tmp_path / "file.csv",
        )