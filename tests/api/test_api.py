from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def valid_payload() -> dict[str, object]:
    """Return one valid API request."""
    return {
        "machine_type": "M",
        "air_temperature": 298.1,
        "process_temperature": 308.6,
        "rotational_speed": 1551,
        "torque": 42.8,
        "tool_wear": 120,
    }


def test_health_endpoint() -> None:
    response = client.get("/health")

    assert response.status_code == 200

    response_data = response.json()

    assert response_data["status"] == "healthy"


def test_invalid_machine_type() -> None:
    payload = valid_payload()
    payload["machine_type"] = "UNKNOWN"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_speed_is_rejected() -> None:
    payload = valid_payload()
    payload["rotational_speed"] = -100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_required_field_is_rejected() -> None:
    payload = valid_payload()
    payload.pop("torque")

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_nan_temperature_is_rejected() -> None:
    payload = valid_payload()
    payload["air_temperature"] = "NaN"

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_extra_field_is_rejected() -> None:
    payload = valid_payload()
    payload["unexpected_feature"] = 100

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422


def test_batch_rejects_empty_machine_list() -> None:
    response = client.post(
        "/predict/batch",
        json={
            "machines": [],
        },
    )

    assert response.status_code == 422
