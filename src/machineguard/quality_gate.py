from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class QualityGateResult:
    """Store the result of model quality-gate evaluation."""

    passed: bool
    checks: dict[str, bool]
    failures: list[str]


def evaluate_quality_gate(
    metrics: dict[str, Any],
    thresholds: dict[str, Any],
) -> QualityGateResult:
    """Evaluate model metrics against configured thresholds.

    Args:
        metrics: Validation metrics produced during training.
        thresholds: Minimum and maximum accepted metric values.

    Returns:
        Quality-gate result containing status and individual checks.
    """
    checks = {
        "minimum_roc_auc": float(metrics["roc_auc"])
        >= float(thresholds["minimum_roc_auc"]),
        "minimum_pr_auc": float(metrics["pr_auc"])
        >= float(thresholds["minimum_pr_auc"]),
        "minimum_recall": float(metrics["recall"])
        >= float(thresholds["minimum_recall"]),
        "minimum_precision": float(metrics["precision"])
        >= float(thresholds["minimum_precision"]),
        "maximum_log_loss": float(metrics["log_loss"])
        <= float(thresholds["maximum_log_loss"]),
        "maximum_business_cost": float(metrics["business_cost"])
        <= float(thresholds["maximum_business_cost"]),
    }

    failures = [check_name for check_name, passed in checks.items() if not passed]

    return QualityGateResult(
        passed=all(checks.values()),
        checks=checks,
        failures=failures,
    )
