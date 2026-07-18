import json
import sys

import pandas as pd
import pandera.pandas as pa

from src.machineguard.config import PROJECT_ROOT, load_config
from src.machineguard.data_schema import validate_machine_data


def main() -> None:
    config = load_config()
    data_path = PROJECT_ROOT / config["data"]["processed_path"]
    report_directory = PROJECT_ROOT / "reports/data_validation"
    report_directory.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(data_path)

    try:
        validated = validate_machine_data(dataframe)

        report = {
            "status": "passed",
            "row_count": len(validated),
            "column_count": len(validated.columns),
            "missing_values": validated.isna().sum().to_dict(),
            "duplicate_rows": int(validated.duplicated().sum()),
            "failure_rate": float(validated["machine_failure"].mean()),
        }

        output_path = report_directory / "validation_report.json"
        output_path.write_text(
            json.dumps(report, indent=2),
            encoding="utf-8",
        )

        print("Dataset validation passed.")
        print(json.dumps(report, indent=2))

    except pa.errors.SchemaErrors as error:
        failure_report = {
            "status": "failed",
            "errors": error.failure_cases.to_dict(orient="records"),
        }

        output_path = report_directory / "validation_report.json"
        output_path.write_text(
            json.dumps(failure_report, indent=2),
            encoding="utf-8",
        )

        print("Dataset validation failed.")
        print(error.failure_cases)
        sys.exit(1)


if __name__ == "__main__":
    main()
