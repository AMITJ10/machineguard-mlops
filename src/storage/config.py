"""Configuration for S3-compatible object storage."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


TRUE_VALUES = {"1", "true", "yes", "on"}
FALSE_VALUES = {"0", "false", "no", "off"}

VALID_PROVIDERS = {"aws", "minio"}
VALID_ADDRESSING_STYLES = {"auto", "path", "virtual"}


def _get_boolean(
    environment: Mapping[str, str],
    key: str,
    default: bool,
) -> bool:
    """Read and validate a boolean environment variable.

    Args:
        environment: Environment-variable mapping.
        key: Environment-variable name.
        default: Value returned when the variable is missing or empty.

    Returns:
        Parsed boolean value.

    Raises:
        ValueError: If the configured value is not a supported boolean.
    """
    raw_value = environment.get(key)

    if raw_value is None or not raw_value.strip():
        return default

    normalized_value = raw_value.strip().lower()

    if normalized_value in TRUE_VALUES:
        return True

    if normalized_value in FALSE_VALUES:
        return False

    raise ValueError(
        f"{key} must be one of "
        f"{sorted(TRUE_VALUES | FALSE_VALUES)}. "
        f"Received: {raw_value!r}",
    )


def _get_optional_string(
    environment: Mapping[str, str],
    key: str,
) -> str | None:
    """Return a stripped environment value or None."""
    value = environment.get(key)

    if value is None:
        return None

    stripped_value = value.strip()
    return stripped_value or None


@dataclass(frozen=True, slots=True)
class S3Settings:
    """Configuration required for an S3-compatible client."""

    bucket_name: str
    region: str
    endpoint_url: str | None
    access_key_id: str | None
    secret_access_key: str | None
    session_token: str | None
    use_ssl: bool
    verify_ssl: bool
    addressing_style: str
    provider: str

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> "S3Settings":
        """Build and validate settings from environment variables.

        Args:
            environment: Optional environment-variable mapping. When omitted,
                ``os.environ`` is used.

        Returns:
            Validated S3 settings.

        Raises:
            ValueError: If mandatory configuration is missing or invalid.
        """
        env = environment if environment is not None else os.environ

        bucket_name = env.get("S3_BUCKET_NAME", "").strip()

        if not bucket_name:
            raise ValueError("S3_BUCKET_NAME must be configured.")

        provider = env.get("STORAGE_PROVIDER", "aws").strip().lower()

        if provider not in VALID_PROVIDERS:
            raise ValueError(
                "STORAGE_PROVIDER must be either 'aws' or 'minio'. "
                f"Received: {provider!r}",
            )

        region = env.get("S3_REGION", "ap-south-1").strip()

        if not region:
            raise ValueError("S3_REGION cannot be empty.")

        endpoint_url = _get_optional_string(env, "S3_ENDPOINT_URL")
        access_key_id = _get_optional_string(env, "AWS_ACCESS_KEY_ID")
        secret_access_key = _get_optional_string(
            env,
            "AWS_SECRET_ACCESS_KEY",
        )
        session_token = _get_optional_string(env, "AWS_SESSION_TOKEN")

        default_addressing_style = (
            "path" if provider == "minio" else "auto"
        )

        addressing_style = env.get(
            "S3_ADDRESSING_STYLE",
            default_addressing_style,
        ).strip().lower()

        if addressing_style not in VALID_ADDRESSING_STYLES:
            raise ValueError(
                "S3_ADDRESSING_STYLE must be one of "
                f"{sorted(VALID_ADDRESSING_STYLES)}. "
                f"Received: {addressing_style!r}",
            )

        use_ssl = _get_boolean(
            environment=env,
            key="S3_USE_SSL",
            default=provider == "aws",
        )

        verify_ssl = _get_boolean(
            environment=env,
            key="S3_VERIFY_SSL",
            default=provider == "aws",
        )

        if provider == "minio":
            if not endpoint_url:
                raise ValueError(
                    "S3_ENDPOINT_URL must be configured when "
                    "STORAGE_PROVIDER=minio.",
                )

            if not access_key_id:
                raise ValueError(
                    "AWS_ACCESS_KEY_ID must be configured for MinIO.",
                )

            if not secret_access_key:
                raise ValueError(
                    "AWS_SECRET_ACCESS_KEY must be configured for MinIO.",
                )

        return cls(
            bucket_name=bucket_name,
            region=region,
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            session_token=session_token,
            use_ssl=use_ssl,
            verify_ssl=verify_ssl,
            addressing_style=addressing_style,
            provider=provider,
        )

    @property
    def is_minio(self) -> bool:
        """Return whether the selected provider is MinIO."""
        return self.provider == "minio"

    @property
    def is_aws(self) -> bool:
        """Return whether the selected provider is AWS."""
        return self.provider == "aws"

    def sanitized_summary(self) -> dict[str, str | bool | None]:
        """Return configuration that is safe to display in logs.

        Secret values are deliberately excluded.
        """
        return {
            "provider": self.provider,
            "bucket_name": self.bucket_name,
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "use_ssl": self.use_ssl,
            "verify_ssl": self.verify_ssl,
            "addressing_style": self.addressing_style,
            "credentials_configured": bool(
                self.access_key_id and self.secret_access_key,
            ),
        }