"""Reusable S3-compatible storage client."""

from __future__ import annotations

import hashlib
import json
import logging
import mimetypes
from collections.abc import Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.client import BaseClient
from botocore.config import Config
from botocore.exceptions import (
    BotoCoreError,
    ClientError,
    EndpointConnectionError,
)

from src.storage.config import S3Settings
from src.storage.exceptions import (
    BucketOperationError,
    ObjectDeleteError,
    ObjectDownloadError,
    ObjectListingError,
    ObjectNotFoundError,
    ObjectUploadError,
)


LOGGER = logging.getLogger(__name__)

NOT_FOUND_ERROR_CODES = {
    "404",
    "NoSuchKey",
    "NotFound",
    "NoSuchBucket",
}

DEFAULT_MULTIPART_THRESHOLD = 8 * 1024 * 1024
DEFAULT_MULTIPART_CHUNKSIZE = 8 * 1024 * 1024
DEFAULT_MAX_CONCURRENCY = 4


def normalize_object_key(object_key: str) -> str:
    """Normalize an S3 object key.

    Args:
        object_key: Raw object key.

    Returns:
        Normalized object key using forward slashes.

    Raises:
        ValueError: If the resulting key is empty.
    """
    normalized_key = object_key.replace("\\", "/").strip().lstrip("/")

    while "//" in normalized_key:
        normalized_key = normalized_key.replace("//", "/")

    if not normalized_key:
        raise ValueError("Object key cannot be empty.")

    return normalized_key


def calculate_sha256(file_path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a local file."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File does not exist: {path}")

    digest = hashlib.sha256()

    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


class S3StorageClient:
    """High-level client for AWS S3 and MinIO."""

    def __init__(
        self,
        settings: S3Settings,
        boto3_client: BaseClient | None = None,
    ) -> None:
        """Initialize the storage client.

        Args:
            settings: Validated S3 configuration.
            boto3_client: Optional pre-created client, primarily for tests.
        """
        self.settings = settings
        self.bucket_name = settings.bucket_name
        self._client = boto3_client or self._build_client()
        self._transfer_config = TransferConfig(
            multipart_threshold=DEFAULT_MULTIPART_THRESHOLD,
            multipart_chunksize=DEFAULT_MULTIPART_CHUNKSIZE,
            max_concurrency=DEFAULT_MAX_CONCURRENCY,
            use_threads=True,
        )

    @classmethod
    def from_environment(cls) -> "S3StorageClient":
        """Create a client directly from environment variables."""
        return cls(S3Settings.from_environment())

    def _build_client(self) -> BaseClient:
        """Create the underlying Boto3 S3 client."""
        client_arguments: dict[str, Any] = {
            "service_name": "s3",
            "region_name": self.settings.region,
            "use_ssl": self.settings.use_ssl,
            "verify": self.settings.verify_ssl,
            "config": Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": (
                        self.settings.addressing_style
                    ),
                },
                retries={
                    "max_attempts": 5,
                    "mode": "standard",
                },
                connect_timeout=10,
                read_timeout=60,
            ),
        }

        if self.settings.endpoint_url:
            client_arguments["endpoint_url"] = (
                self.settings.endpoint_url
            )

        if self.settings.access_key_id:
            client_arguments["aws_access_key_id"] = (
                self.settings.access_key_id
            )

        if self.settings.secret_access_key:
            client_arguments["aws_secret_access_key"] = (
                self.settings.secret_access_key
            )

        if self.settings.session_token:
            client_arguments["aws_session_token"] = (
                self.settings.session_token
            )

        return boto3.client(**client_arguments)

    @property
    def client(self) -> BaseClient:
        """Expose the underlying Boto3 client."""
        return self._client

    def bucket_exists(self) -> bool:
        """Check whether the configured bucket is accessible."""
        try:
            self._client.head_bucket(Bucket=self.bucket_name)
            return True
        except ClientError as error:
            error_code = self._extract_error_code(error)

            if error_code in NOT_FOUND_ERROR_CODES:
                return False

            raise BucketOperationError(
                f"Could not inspect bucket {self.bucket_name!r}: "
                f"{error}",
            ) from error
        except (BotoCoreError, EndpointConnectionError) as error:
            raise BucketOperationError(
                f"Could not connect to object storage: {error}",
            ) from error

    def create_bucket(self) -> bool:
        """Create the configured bucket if it does not exist.

        Returns:
            True when a new bucket was created. False when it already existed.
        """
        if self.bucket_exists():
            LOGGER.info(
                "Bucket already exists: %s",
                self.bucket_name,
            )
            return False

        create_arguments: dict[str, Any] = {
            "Bucket": self.bucket_name,
        }

        if (
            self.settings.provider == "aws"
            and self.settings.region != "us-east-1"
        ):
            create_arguments["CreateBucketConfiguration"] = {
                "LocationConstraint": self.settings.region,
            }

        try:
            self._client.create_bucket(**create_arguments)

            waiter = self._client.get_waiter("bucket_exists")
            waiter.wait(Bucket=self.bucket_name)

            LOGGER.info(
                "Created bucket: %s",
                self.bucket_name,
            )
            return True
        except (ClientError, BotoCoreError) as error:
            raise BucketOperationError(
                f"Could not create bucket {self.bucket_name!r}: "
                f"{error}",
            ) from error

    def initialize_prefixes(
        self,
        prefixes: list[str],
    ) -> list[str]:
        """Create placeholder objects for logical bucket prefixes.

        S3 does not have real directories. A trailing-slash placeholder object
        makes the prefixes visible in storage consoles.

        Args:
            prefixes: Prefix names to initialize.

        Returns:
            List of initialized prefix keys.
        """
        initialized_prefixes: list[str] = []

        for prefix in prefixes:
            normalized_prefix = normalize_object_key(prefix).rstrip("/") + "/"

            try:
                self._client.put_object(
                    Bucket=self.bucket_name,
                    Key=normalized_prefix,
                    Body=b"",
                    ContentType="application/x-directory",
                )
                initialized_prefixes.append(normalized_prefix)
            except (ClientError, BotoCoreError) as error:
                raise BucketOperationError(
                    f"Could not initialize prefix "
                    f"{normalized_prefix!r}: {error}",
                ) from error

        return initialized_prefixes

    def upload_file(
        self,
        local_path: str | Path,
        object_key: str,
        *,
        metadata: Mapping[str, str] | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        """Upload a local file.

        Args:
            local_path: Local file to upload.
            object_key: Destination key in the bucket.
            metadata: Optional string metadata.
            content_type: Optional MIME type override.

        Returns:
            Upload information containing bucket, key, size and checksum.
        """
        source_path = Path(local_path)

        if not source_path.is_file():
            raise FileNotFoundError(
                f"Local upload file does not exist: {source_path}",
            )

        normalized_key = normalize_object_key(object_key)
        checksum = calculate_sha256(source_path)

        detected_content_type = (
            content_type
            or mimetypes.guess_type(source_path.name)[0]
            or "application/octet-stream"
        )

        upload_metadata = {
            "sha256": checksum,
            "uploaded-at": datetime.now(timezone.utc).isoformat(),
        }

        if metadata:
            upload_metadata.update(
                {
                    str(key): str(value)
                    for key, value in metadata.items()
                },
            )

        extra_arguments = {
            "ContentType": detected_content_type,
            "Metadata": upload_metadata,
        }

        try:
            self._client.upload_file(
                Filename=str(source_path),
                Bucket=self.bucket_name,
                Key=normalized_key,
                ExtraArgs=extra_arguments,
                Config=self._transfer_config,
            )

            LOGGER.info(
                "Uploaded %s to s3://%s/%s",
                source_path,
                self.bucket_name,
                normalized_key,
            )

            return {
                "bucket": self.bucket_name,
                "key": normalized_key,
                "size_bytes": source_path.stat().st_size,
                "sha256": checksum,
                "content_type": detected_content_type,
                "uri": self.build_uri(normalized_key),
            }
        except (ClientError, BotoCoreError, OSError) as error:
            raise ObjectUploadError(
                f"Could not upload {source_path} to "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def upload_fileobj(
        self,
        file_object: BinaryIO,
        object_key: str,
        *,
        metadata: Mapping[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ) -> str:
        """Upload an already-open binary file object."""
        normalized_key = normalize_object_key(object_key)

        extra_arguments: dict[str, Any] = {
            "ContentType": content_type,
        }

        if metadata:
            extra_arguments["Metadata"] = {
                str(key): str(value)
                for key, value in metadata.items()
            }

        try:
            self._client.upload_fileobj(
                Fileobj=file_object,
                Bucket=self.bucket_name,
                Key=normalized_key,
                ExtraArgs=extra_arguments,
                Config=self._transfer_config,
            )
            return self.build_uri(normalized_key)
        except (ClientError, BotoCoreError, OSError) as error:
            raise ObjectUploadError(
                f"Could not upload file object to "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def upload_json(
        self,
        data: Mapping[str, Any] | list[Any],
        object_key: str,
        *,
        metadata: Mapping[str, str] | None = None,
    ) -> str:
        """Serialize and upload a JSON object."""
        normalized_key = normalize_object_key(object_key)
        serialized_data = json.dumps(
            data,
            indent=2,
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")

        extra_arguments: dict[str, Any] = {
            "ContentType": "application/json",
        }

        if metadata:
            extra_arguments["Metadata"] = {
                str(key): str(value)
                for key, value in metadata.items()
            }

        try:
            self._client.put_object(
                Bucket=self.bucket_name,
                Key=normalized_key,
                Body=serialized_data,
                **extra_arguments,
            )
            return self.build_uri(normalized_key)
        except (ClientError, BotoCoreError) as error:
            raise ObjectUploadError(
                f"Could not upload JSON object "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def download_file(
        self,
        object_key: str,
        local_path: str | Path,
        *,
        create_parent_directories: bool = True,
    ) -> Path:
        """Download an object to a local file."""
        normalized_key = normalize_object_key(object_key)
        destination_path = Path(local_path)

        if create_parent_directories:
            destination_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        try:
            self._client.download_file(
                Bucket=self.bucket_name,
                Key=normalized_key,
                Filename=str(destination_path),
                Config=self._transfer_config,
            )

            LOGGER.info(
                "Downloaded s3://%s/%s to %s",
                self.bucket_name,
                normalized_key,
                destination_path,
            )
            return destination_path
        except ClientError as error:
            error_code = self._extract_error_code(error)

            if error_code in NOT_FOUND_ERROR_CODES:
                raise ObjectNotFoundError(
                    f"Object does not exist: "
                    f"s3://{self.bucket_name}/{normalized_key}",
                ) from error

            raise ObjectDownloadError(
                f"Could not download "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error
        except (BotoCoreError, OSError) as error:
            raise ObjectDownloadError(
                f"Could not download "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def read_bytes(self, object_key: str) -> bytes:
        """Read an object's contents into memory."""
        normalized_key = normalize_object_key(object_key)

        try:
            response = self._client.get_object(
                Bucket=self.bucket_name,
                Key=normalized_key,
            )
            return response["Body"].read()
        except ClientError as error:
            error_code = self._extract_error_code(error)

            if error_code in NOT_FOUND_ERROR_CODES:
                raise ObjectNotFoundError(
                    f"Object does not exist: "
                    f"s3://{self.bucket_name}/{normalized_key}",
                ) from error

            raise ObjectDownloadError(
                f"Could not read "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error
        except BotoCoreError as error:
            raise ObjectDownloadError(
                f"Could not read "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def read_json(self, object_key: str) -> Any:
        """Read and deserialize a JSON object."""
        raw_content = self.read_bytes(object_key)

        try:
            return json.loads(raw_content.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ObjectDownloadError(
                f"Object is not valid UTF-8 JSON: {object_key}",
            ) from error

    def object_exists(self, object_key: str) -> bool:
        """Check whether an object exists."""
        normalized_key = normalize_object_key(object_key)

        try:
            self._client.head_object(
                Bucket=self.bucket_name,
                Key=normalized_key,
            )
            return True
        except ClientError as error:
            error_code = self._extract_error_code(error)

            if error_code in NOT_FOUND_ERROR_CODES:
                return False

            raise ObjectDownloadError(
                f"Could not inspect "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error
        except BotoCoreError as error:
            raise ObjectDownloadError(
                f"Could not inspect "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def get_object_metadata(
        self,
        object_key: str,
    ) -> dict[str, Any]:
        """Retrieve metadata for one object."""
        normalized_key = normalize_object_key(object_key)

        try:
            response = self._client.head_object(
                Bucket=self.bucket_name,
                Key=normalized_key,
            )

            return {
                "bucket": self.bucket_name,
                "key": normalized_key,
                "content_length": response.get("ContentLength"),
                "content_type": response.get("ContentType"),
                "etag": str(response.get("ETag", "")).strip('"'),
                "last_modified": response.get("LastModified"),
                "metadata": response.get("Metadata", {}),
                "version_id": response.get("VersionId"),
            }
        except ClientError as error:
            error_code = self._extract_error_code(error)

            if error_code in NOT_FOUND_ERROR_CODES:
                raise ObjectNotFoundError(
                    f"Object does not exist: "
                    f"s3://{self.bucket_name}/{normalized_key}",
                ) from error

            raise ObjectDownloadError(
                f"Could not read object metadata for "
                f"{normalized_key!r}: {error}",
            ) from error
        except BotoCoreError as error:
            raise ObjectDownloadError(
                f"Could not read object metadata for "
                f"{normalized_key!r}: {error}",
            ) from error

    def iter_objects(
        self,
        prefix: str = "",
    ) -> Iterator[dict[str, Any]]:
        """Iterate over all objects under a prefix."""
        normalized_prefix = (
            normalize_object_key(prefix)
            if prefix.strip()
            else ""
        )

        paginator = self._client.get_paginator("list_objects_v2")

        try:
            pages = paginator.paginate(
                Bucket=self.bucket_name,
                Prefix=normalized_prefix,
            )

            for page in pages:
                for object_info in page.get("Contents", []):
                    yield {
                        "key": object_info["Key"],
                        "size_bytes": object_info["Size"],
                        "etag": str(
                            object_info.get("ETag", ""),
                        ).strip('"'),
                        "last_modified": object_info.get(
                            "LastModified",
                        ),
                        "storage_class": object_info.get(
                            "StorageClass",
                        ),
                    }
        except (ClientError, BotoCoreError) as error:
            raise ObjectListingError(
                f"Could not list objects from bucket "
                f"{self.bucket_name!r}: {error}",
            ) from error

    def list_objects(
        self,
        prefix: str = "",
    ) -> list[dict[str, Any]]:
        """Return all objects under a prefix as a list."""
        return list(self.iter_objects(prefix=prefix))

    def delete_object(self, object_key: str) -> None:
        """Delete one object."""
        normalized_key = normalize_object_key(object_key)

        try:
            self._client.delete_object(
                Bucket=self.bucket_name,
                Key=normalized_key,
            )
        except (ClientError, BotoCoreError) as error:
            raise ObjectDeleteError(
                f"Could not delete "
                f"s3://{self.bucket_name}/{normalized_key}: {error}",
            ) from error

    def delete_prefix(self, prefix: str) -> int:
        """Delete all objects under a prefix.

        Returns:
            Number of requested object deletions.
        """
        object_keys = [
            item["key"]
            for item in self.iter_objects(prefix=prefix)
        ]

        deleted_count = 0

        for batch_start in range(0, len(object_keys), 1000):
            batch = object_keys[batch_start : batch_start + 1000]

            if not batch:
                continue

            try:
                response = self._client.delete_objects(
                    Bucket=self.bucket_name,
                    Delete={
                        "Objects": [
                            {"Key": object_key}
                            for object_key in batch
                        ],
                        "Quiet": True,
                    },
                )

                errors = response.get("Errors", [])

                if errors:
                    raise ObjectDeleteError(
                        f"Some objects could not be deleted: {errors}",
                    )

                deleted_count += len(batch)
            except (ClientError, BotoCoreError) as error:
                raise ObjectDeleteError(
                    f"Could not delete prefix {prefix!r}: {error}",
                ) from error

        return deleted_count

    def sync_directory(
        self,
        local_directory: str | Path,
        destination_prefix: str,
        *,
        allowed_extensions: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Upload all files from a local directory recursively.

        Args:
            local_directory: Directory to synchronize.
            destination_prefix: Destination bucket prefix.
            allowed_extensions: Optional lowercase extension filter, such as
                ``{".json", ".csv"}``.

        Returns:
            Upload result for every synchronized file.
        """
        source_directory = Path(local_directory)

        if not source_directory.is_dir():
            raise NotADirectoryError(
                f"Directory does not exist: {source_directory}",
            )

        normalized_prefix = normalize_object_key(
            destination_prefix,
        ).rstrip("/")

        normalized_extensions = (
            {extension.lower() for extension in allowed_extensions}
            if allowed_extensions
            else None
        )

        upload_results: list[dict[str, Any]] = []

        for local_file in sorted(source_directory.rglob("*")):
            if not local_file.is_file():
                continue

            if (
                normalized_extensions is not None
                and local_file.suffix.lower()
                not in normalized_extensions
            ):
                continue

            relative_path = local_file.relative_to(source_directory)
            object_key = (
                f"{normalized_prefix}/"
                f"{relative_path.as_posix()}"
            )

            upload_results.append(
                self.upload_file(
                    local_path=local_file,
                    object_key=object_key,
                    metadata={
                        "source-directory": str(source_directory),
                    },
                ),
            )

        return upload_results

    def build_uri(self, object_key: str) -> str:
        """Build the canonical S3 URI for an object."""
        normalized_key = normalize_object_key(object_key)

        return f"s3://{self.bucket_name}/{normalized_key}"

    @staticmethod
    def _extract_error_code(error: ClientError) -> str:
        """Extract the AWS error code from a ClientError."""
        return str(
            error.response.get("Error", {}).get("Code", ""),
        )