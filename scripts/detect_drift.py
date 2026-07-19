"""Run MachineGuard data-drift detection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.monitoring import (
    load_yaml_config,
    run_drift_detection,
)


def resolve_project_path(
    path_value: str,
) -> Path:
    """Resolve a configuration path relative to the project root."""

    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Detect feature drift in MachineGuard data."
        )
    )

    parser.add_argument(
        "--config",
        default="configs/drift_config.yaml",
        help="Path to the drift YAML configuration.",
    )

    parser.add_argument(
        "--reference-data",
        default=None,
        help="Optional reference CSV override.",
    )

    parser.add_argument(
        "--current-data",
        default=None,
        help="Optional current CSV override.",
    )

    return parser.parse_args()


def main() -> int:
    """Run drift detection and print its summary."""

    arguments = parse_arguments()

    config_path = resolve_project_path(
        arguments.config
    )

    config = load_yaml_config(
        config_path
    )

    drift_config: dict[str, Any] = config["drift"]
    path_config: dict[str, str] = config["paths"]

    reference_path = resolve_project_path(
        arguments.reference_data
        or path_config["reference_data"]
    )

    current_path = resolve_project_path(
        arguments.current_data
        or path_config["current_data"]
    )

    result = run_drift_detection(
        reference_path=reference_path,
        current_path=current_path,
        json_report_path=resolve_project_path(
            path_config["drift_json_report"]
        ),
        html_report_path=resolve_project_path(
            path_config["drift_html_report"]
        ),
        summary_path=resolve_project_path(
            path_config["drift_summary"]
        ),
        excluded_columns=drift_config.get(
            "excluded_columns",
            [],
        ),
        drift_share_threshold=float(
            drift_config.get(
                "drift_share_threshold",
                0.30,
            )
        ),
        method=drift_config.get(
            "method"
        ),
        significance_level=float(
            drift_config.get(
                "significance_level",
                0.05,
            )
        ),
        numeric_effect_size_threshold=float(
            drift_config.get(
                "numeric_effect_size_threshold",
                0.10,
            )
        ),
        categorical_effect_size_threshold=float(
            drift_config.get(
                "categorical_effect_size_threshold",
                0.10,
            )
        ),
    )

    print(
        json.dumps(
            result.to_dict(),
            indent=2,
        )
    )

    if (
        result.dataset_drift_detected
        and drift_config.get(
            "fail_on_drift",
            False,
        )
    ):
        print(
            "Data drift exceeded the configured threshold."
        )

        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )