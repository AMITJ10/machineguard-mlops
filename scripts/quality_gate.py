from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]

QUALITY_GATE_RESULT_PATH = (
    PROJECT_ROOT / "reports" / "quality_gate" / "quality_gate_result.json"
)


def save_quality_gate_result(
    result: dict[str, Any],
) -> None:
    """Save the model quality-gate decision.

    Args:
        result: Quality-gate metrics and approval result.
    """
    QUALITY_GATE_RESULT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    QUALITY_GATE_RESULT_PATH.write_text(
        json.dumps(
            result,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
