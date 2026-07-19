"""
Export the current Champion model from MLflow.

This script is executed after a successful training run.

Output:

artifacts/
    production/
        model/
"""

from pathlib import Path

import mlflow
from mlflow import MlflowClient


MODEL_NAME = "MachineGuardFailureClassifier"
MODEL_ALIAS = "champion"

OUTPUT_DIR = Path("artifacts/production/model")


def main():

    client = MlflowClient()

    version = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS,
    )

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    print(f"Loading {model_uri}")

    model = mlflow.pyfunc.load_model(model_uri)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.pyfunc.save_model(
        path=str(OUTPUT_DIR),
        python_model=model._model_impl.python_model
        if hasattr(model._model_impl, "python_model")
        else None,
        artifacts=None,
    )

    print()

    print("=" * 60)
    print("Champion model exported.")
    print(OUTPUT_DIR.resolve())
    print("=" * 60)


if __name__ == "__main__":
    main()