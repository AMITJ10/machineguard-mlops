from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


MachineType = Literal["L", "M", "H"]

RiskLevel = Literal[
    "low",
    "medium",
    "high",
    "critical",
]


class MachinePredictionRequest(BaseModel):
    """Validated machine measurements submitted for prediction."""

    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "machine_type": "M",
                "air_temperature": 298.1,
                "process_temperature": 308.6,
                "rotational_speed": 1551.0,
                "torque": 42.8,
                "tool_wear": 0.0,
            }
        },
    )

    machine_type: MachineType = Field(
        ...,
        description=(
            "Machine type: L for low quality, "
            "M for medium quality or H for high quality."
        ),
    )

    air_temperature: Annotated[
        float,
        Field(
            ge=250.0,
            le=400.0,
            description="Air temperature in Kelvin.",
        ),
    ]

    process_temperature: Annotated[
        float,
        Field(
            ge=250.0,
            le=450.0,
            description="Process temperature in Kelvin.",
        ),
    ]

    rotational_speed: Annotated[
        float,
        Field(
            ge=0.0,
            le=5000.0,
            description="Rotational speed in revolutions per minute.",
        ),
    ]

    torque: Annotated[
        float,
        Field(
            ge=0.0,
            le=200.0,
            description="Torque in Newton-metres.",
        ),
    ]

    tool_wear: Annotated[
        float,
        Field(
            ge=0.0,
            le=500.0,
            description="Accumulated tool wear in minutes.",
        ),
    ]

    @field_validator(
        "machine_type",
        mode="before",
    )
    @classmethod
    def normalize_machine_type(
        cls,
        value: object,
    ) -> object:
        """Normalize the machine type before validation."""
        if isinstance(value, str):
            return value.strip().upper()

        return value

    @field_validator(
        "air_temperature",
        "process_temperature",
        "rotational_speed",
        "torque",
        "tool_wear",
    )
    @classmethod
    def reject_non_finite_values(
        cls,
        value: float,
    ) -> float:
        """Reject NaN and infinite numeric measurements."""
        if value != value:
            raise ValueError("Numeric values must not be NaN.")

        if value in {
            float("inf"),
            float("-inf"),
        }:
            raise ValueError("Numeric values must be finite.")

        return value


class MachinePredictionResponse(BaseModel):
    """Prediction returned for one machine."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction_id": ("6e590a87-3c05-47ee-bcc2-173fd30e5151"),
                "prediction": 0,
                "failure_probability": 0.084216,
                "risk_level": "low",
                "threshold": 0.5,
                "model_name": ("MachineGuardFailureClassifier"),
                "model_alias": "champion",
                "model_version": "1",
            }
        }
    )

    prediction_id: UUID

    prediction: Literal[0, 1] = Field(
        ...,
        description=(
            "One indicates predicted failure; zero indicates no predicted failure."
        ),
    )

    failure_probability: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
            description="Estimated probability of machine failure.",
        ),
    ]

    risk_level: RiskLevel

    threshold: Annotated[
        float,
        Field(
            ge=0.0,
            le=1.0,
        ),
    ]

    model_name: str = Field(
        ...,
        min_length=1,
    )

    model_alias: str = Field(
        ...,
        min_length=1,
    )

    model_version: str = Field(
        ...,
        min_length=1,
    )


class BatchPredictionRequest(BaseModel):
    """Request containing multiple machines."""

    model_config = ConfigDict(
        extra="forbid",
    )

    machines: Annotated[
        list[MachinePredictionRequest],
        Field(
            min_length=1,
            max_length=1000,
            description=("Machines to score. The maximum batch size is 1000."),
        ),
    ]


class BatchPredictionResponse(BaseModel):
    """Prediction response for multiple machines."""

    predictions: list[MachinePredictionResponse]

    total_predictions: Annotated[
        int,
        Field(ge=0),
    ]

    predicted_failures: Annotated[
        int,
        Field(ge=0),
    ]


class RootResponse(BaseModel):
    """Information returned by the API root endpoint."""

    application: str

    version: str

    documentation: str

    health: str

    readiness: str

    metrics: str


class HealthResponse(BaseModel):
    """Health-check response."""

    status: Literal["healthy", "unhealthy"]

    service: str

    timestamp: str


class ReadinessResponse(BaseModel):
    """Model-readiness response."""

    status: Literal["ready", "not_ready"]

    model_loaded: bool

    model_name: str

    model_alias: str

    model_version: str | None = None


class ErrorResponse(BaseModel):
    """Standard structured error response."""

    error: str

    message: str

    details: dict[str, object] | None = None
