"""Custom exceptions for object-storage operations."""

from __future__ import annotations


class StorageError(RuntimeError):
    """Base exception for all storage-related failures."""


class StorageConfigurationError(StorageError):
    """Raised when object-storage configuration is invalid."""


class BucketOperationError(StorageError):
    """Raised when a bucket operation fails."""


class ObjectUploadError(StorageError):
    """Raised when an object upload fails."""


class ObjectDownloadError(StorageError):
    """Raised when an object download fails."""


class ObjectNotFoundError(StorageError):
    """Raised when a requested object does not exist."""


class ObjectDeleteError(StorageError):
    """Raised when an object deletion fails."""


class ObjectListingError(StorageError):
    """Raised when listing objects fails."""