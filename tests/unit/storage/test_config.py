"""Unit tests for object-storage configuration."""

from __future__ import annotations

import pytest

from src.storage.config import S3Settings, _get_boolean


def test_get_boolean_returns_default_when_missing() -> None:
    """Missing boolean values should use the provided default."""
    assert _get_boolean({}, "EXAMPLE", True) is True
    assert _get_boolean({}, "EXAMPLE", False) is False


@pytest.mark.parametrize(
    "raw_value",
    ["1", "true", "TRUE", "yes", "YES", "on"],
)
def test_get_boolean_parses_true_values(
    raw_value: str,
) -> None:
    """Supported true values should return True."""
    assert _get_boolean(
        {"EXAMPLE": raw_value},
        "EXAMPLE",
        False,
    ) is True


@pytest.mark.parametrize(
    "raw_value",
    ["0", "false", "FALSE", "no", "NO", "off"],
)
def test_get_boolean_parses_false_values(
    raw_value: str,
) -> None:
    """Supported false values should return False."""
    assert _get_boolean(
        {"EXAMPLE": raw_value},
        "EXAMPLE",
        True,
    ) is False


def test_get_boolean_rejects_invalid_value() -> None:
    """Unexpected boolean values should fail validation."""
    with pytest.raises(ValueError, match="EXAMPLE must be one of"):
        _get_boolean(
            {"EXAMPLE": "sometimes"},
            "EXAMPLE",
            False,
        )


def test_minio_settings_are_loaded() -> None:
    """Complete MinIO settings should be parsed successfully."""
    settings = S3Settings.from_environment(
        {
            "STORAGE_PROVIDER": "minio",
            "S3_BUCKET_NAME": "test-bucket",
            "S3_REGION": "ap-south-1",
            "S3_ENDPOINT_URL": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "test-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret",
            "S3_USE_SSL": "false",
            "S3_VERIFY_SSL": "false",
            "S3_ADDRESSING_STYLE": "path",
        },
    )

    assert settings.provider == "minio"
    assert settings.bucket_name == "test-bucket"
    assert settings.endpoint_url == "http://localhost:9000"
    assert settings.use_ssl is False
    assert settings.verify_ssl is False
    assert settings.addressing_style == "path"
    assert settings.is_minio is True
    assert settings.is_aws is False


def test_aws_settings_can_use_iam_role_credentials() -> None:
    """Explicit credentials should not be mandatory for AWS."""
    settings = S3Settings.from_environment(
        {
            "STORAGE_PROVIDER": "aws",
            "S3_BUCKET_NAME": "test-bucket",
            "S3_REGION": "ap-south-1",
        },
    )

    assert settings.provider == "aws"
    assert settings.endpoint_url is None
    assert settings.access_key_id is None
    assert settings.secret_access_key is None
    assert settings.use_ssl is True
    assert settings.verify_ssl is True
    assert settings.addressing_style == "auto"


def test_bucket_name_is_required() -> None:
    """Empty bucket names should fail validation."""
    with pytest.raises(
        ValueError,
        match="S3_BUCKET_NAME must be configured",
    ):
        S3Settings.from_environment(
            {
                "STORAGE_PROVIDER": "aws",
            },
        )


def test_invalid_provider_is_rejected() -> None:
    """Only AWS and MinIO should be supported."""
    with pytest.raises(
        ValueError,
        match="STORAGE_PROVIDER must be either",
    ):
        S3Settings.from_environment(
            {
                "STORAGE_PROVIDER": "azure",
                "S3_BUCKET_NAME": "test-bucket",
            },
        )


def test_minio_requires_endpoint() -> None:
    """MinIO must have a reachable endpoint."""
    with pytest.raises(
        ValueError,
        match="S3_ENDPOINT_URL must be configured",
    ):
        S3Settings.from_environment(
            {
                "STORAGE_PROVIDER": "minio",
                "S3_BUCKET_NAME": "test-bucket",
                "AWS_ACCESS_KEY_ID": "test-key",
                "AWS_SECRET_ACCESS_KEY": "test-secret",
            },
        )


def test_minio_requires_access_key() -> None:
    """MinIO must have an access key."""
    with pytest.raises(
        ValueError,
        match="AWS_ACCESS_KEY_ID must be configured",
    ):
        S3Settings.from_environment(
            {
                "STORAGE_PROVIDER": "minio",
                "S3_BUCKET_NAME": "test-bucket",
                "S3_ENDPOINT_URL": "http://localhost:9000",
                "AWS_SECRET_ACCESS_KEY": "test-secret",
            },
        )


def test_invalid_addressing_style_is_rejected() -> None:
    """Only supported Boto3 addressing styles should be allowed."""
    with pytest.raises(
        ValueError,
        match="S3_ADDRESSING_STYLE must be one of",
    ):
        S3Settings.from_environment(
            {
                "STORAGE_PROVIDER": "aws",
                "S3_BUCKET_NAME": "test-bucket",
                "S3_ADDRESSING_STYLE": "invalid",
            },
        )


def test_sanitized_summary_does_not_expose_secrets() -> None:
    """Configuration summaries must not expose secret credentials."""
    settings = S3Settings.from_environment(
        {
            "STORAGE_PROVIDER": "minio",
            "S3_BUCKET_NAME": "test-bucket",
            "S3_ENDPOINT_URL": "http://localhost:9000",
            "AWS_ACCESS_KEY_ID": "sensitive-key",
            "AWS_SECRET_ACCESS_KEY": "sensitive-secret",
        },
    )

    summary = settings.sanitized_summary()
    serialized_summary = str(summary)

    assert "sensitive-key" not in serialized_summary
    assert "sensitive-secret" not in serialized_summary
    assert summary["credentials_configured"] is True