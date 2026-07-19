"""Object-storage utilities for MachineGuard."""

from src.storage.config import S3Settings
from src.storage.exceptions import (
    BucketOperationError,
    ObjectDeleteError,
    ObjectDownloadError,
    ObjectListingError,
    ObjectNotFoundError,
    ObjectUploadError,
    StorageConfigurationError,
    StorageError,
)
from src.storage.s3_client import (
    S3StorageClient,
    calculate_sha256,
    normalize_object_key,
)

__all__ = [
    "BucketOperationError",
    "ObjectDeleteError",
    "ObjectDownloadError",
    "ObjectListingError",
    "ObjectNotFoundError",
    "ObjectUploadError",
    "S3Settings",
    "S3StorageClient",
    "StorageConfigurationError",
    "StorageError",
    "calculate_sha256",
    "normalize_object_key",
]