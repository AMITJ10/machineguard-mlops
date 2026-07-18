from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)
from sklearn.pipeline import Pipeline

from src.machineguard.config import PROJECT_ROOT, load_config
from src.machineguard.data_loader import (
    load_processed_data,
    split_machine_data,
)
from src.machineguard.data_schema import validate_machine_data
from src.machineguard.mlflow_utils import configure_mlflow
from src.machineguard.evaluation import evaluate_classifier
from src.machineguard.pipeline import build_training_pipeline


def save_json(
    data: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a dictionary as formatted JSON.

    Args:
        data: Dictionary to save.
        output_path: Destination JSON file path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(
            data,
            indent=2,
        ),
        encoding="utf-8",
    )


def save_confusion_matrix(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save the confusion-matrix figure.

    Args:
        model: Trained Scikit-learn pipeline.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        labels=[0, 1],
        display_labels=[
            "No Failure",
            "Failure",
        ],
    )

    display.ax_.set_title("Machine Failure Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def save_roc_curve(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save the ROC curve.

    Args:
        model: Trained Scikit-learn pipeline.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = RocCurveDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    display.ax_.set_title("Machine Failure ROC Curve")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def save_precision_recall_curve(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_path: Path,
) -> None:
    """Save the precision-recall curve.

    Args:
        model: Trained Scikit-learn pipeline.
        X_test: Test features.
        y_test: Test labels.
        output_path: Destination image path.
    """
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    display = PrecisionRecallDisplay.from_estimator(
        model,
        X_test,
        y_test,
    )

    display.ax_.set_title("Machine Failure Precision-Recall Curve")

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close()


def main() -> None:
    """Train, evaluate, save, and track the machine-failure model."""
    config = load_config()

    random_state = int(config["project"]["random_state"])

    target_column = str(config["data"]["target_column"])

    processed_data_path = PROJECT_ROOT / config["data"]["processed_path"]

    reference_data_path = PROJECT_ROOT / config["data"]["reference_path"]

    artifacts_directory = PROJECT_ROOT / "artifacts"

    model_path = artifacts_directory / "machineguard_pipeline.joblib"

    test_metrics_path = artifacts_directory / "test_metrics.json"

    validation_metrics_path = artifacts_directory / "validation_metrics.json"

    split_summary_path = artifacts_directory / "split_summary.json"

    figures_directory = PROJECT_ROOT / "reports" / "figures"

    default_tracking_uri = str(config["mlflow"]["tracking_uri"])

    experiment_name = str(config["mlflow"]["experiment_name"])

    registered_model_name = str(config["registry"]["model_name"])

    threshold = float(
        config["model"].get(
            "threshold",
            0.50,
        )
    )

    dataframe = load_processed_data(processed_data_path)

    dataframe = validate_machine_data(dataframe)

    (
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    ) = split_machine_data(
        dataframe=dataframe,
        target_column=target_column,
        test_size=float(config["data"]["test_size"]),
        validation_size=float(config["data"]["validation_size"]),
        random_state=random_state,
    )

    n_estimators = int(config["training"]["n_estimators"])

    max_depth_value = config["training"]["max_depth"]

    max_depth = int(max_depth_value) if max_depth_value is not None else None

    min_samples_split = int(config["training"]["min_samples_split"])

    model = build_training_pipeline(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=random_state,
    )

    tracking_uri, registry_uri = configure_mlflow(
        default_tracking_uri=default_tracking_uri,
    )

    print(f"MLflow tracking URI: {tracking_uri}")
    print(f"MLflow registry URI: {registry_uri}")
    print(f"MLflow experiment: {experiment_name}")
    print(f"Registered model name: {registered_model_name}")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        model.fit(
            X_train,
            y_train,
        )

        validation_probabilities = model.predict_proba(X_validation)[:, 1]

        test_probabilities = model.predict_proba(X_test)[:, 1]

        validation_metrics = evaluate_classifier(
            y_true=y_validation,
            probabilities=validation_probabilities,
            threshold=threshold,
        )

        test_metrics = evaluate_classifier(
            y_true=y_test,
            probabilities=test_probabilities,
            threshold=threshold,
        )

        split_summary = {
            "total_rows": int(len(dataframe)),
            "training_rows": int(len(X_train)),
            "validation_rows": int(len(X_validation)),
            "test_rows": int(len(X_test)),
            "training_failure_rate": float(y_train.mean()),
            "validation_failure_rate": float(y_validation.mean()),
            "test_failure_rate": float(y_test.mean()),
        }

        artifacts_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            model_path,
        )

        save_json(
            validation_metrics,
            validation_metrics_path,
        )

        save_json(
            test_metrics,
            test_metrics_path,
        )

        save_json(
            split_summary,
            split_summary_path,
        )

        reference_dataframe = X_train.copy()

        reference_dataframe[target_column] = y_train.to_numpy()

        reference_data_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        reference_dataframe.to_csv(
            reference_data_path,
            index=False,
        )

        confusion_matrix_path = figures_directory / "confusion_matrix.png"

        roc_curve_path = figures_directory / "roc_curve.png"

        precision_recall_curve_path = figures_directory / "precision_recall_curve.png"

        save_confusion_matrix(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_path=confusion_matrix_path,
        )

        save_roc_curve(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_path=roc_curve_path,
        )

        save_precision_recall_curve(
            model=model,
            X_test=X_test,
            y_test=y_test,
            output_path=precision_recall_curve_path,
        )

        mlflow.log_params(
            {
                "model_type": ("RandomForestClassifier"),
                "n_estimators": n_estimators,
                "max_depth": (max_depth if max_depth is not None else "None"),
                "min_samples_split": (min_samples_split),
                "class_weight": "balanced",
                "threshold": threshold,
                "random_state": random_state,
                "training_rows": len(X_train),
                "validation_rows": len(X_validation),
                "test_rows": len(X_test),
            }
        )

        mlflow.log_metrics(
            {
                f"validation_{name}": float(value)
                for name, value in (validation_metrics.items())
                if isinstance(
                    value,
                    (int, float),
                )
            }
        )

        mlflow.log_metrics(
            {
                f"test_{name}": float(value)
                for name, value in (test_metrics.items())
                if isinstance(
                    value,
                    (int, float),
                )
            }
        )

        mlflow.log_artifact(
            str(test_metrics_path),
            artifact_path="metrics",
        )

        mlflow.log_artifact(
            str(validation_metrics_path),
            artifact_path="metrics",
        )

        mlflow.log_artifact(
            str(split_summary_path),
            artifact_path="data",
        )

        mlflow.log_artifact(
            str(confusion_matrix_path),
            artifact_path="figures",
        )

        mlflow.log_artifact(
            str(roc_curve_path),
            artifact_path="figures",
        )

        mlflow.log_artifact(
            str(precision_recall_curve_path),
            artifact_path="figures",
        )

        input_example = X_train.head(5).copy()

        numeric_input_columns = [
            "air_temperature",
            "process_temperature",
            "rotational_speed",
            "torque",
            "tool_wear",
        ]

        existing_numeric_columns = [
            column
            for column in numeric_input_columns
            if column in input_example.columns
        ]

        input_example[existing_numeric_columns] = input_example[
            existing_numeric_columns
        ].astype("float64")

        prediction_example = model.predict_proba(input_example)

        signature = infer_signature(
            input_example,
            prediction_example,
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            registered_model_name=registered_model_name,
            signature=signature,
            input_example=input_example,
            pyfunc_predict_fn="predict_proba",
            serialization_format="cloudpickle",
            code_paths=[
                str(PROJECT_ROOT / "src"),
            ],
        )

        print("Model training completed successfully.")

        print(f"MLflow run ID: {run.info.run_id}")

        print(f"MLflow model URI: {model_info.model_uri}")
        print(f"Registered model name: {registered_model_name}")

        if model_info.registered_model_version is not None:
            print(f"Registered model version: {model_info.registered_model_version}")
        print(f"Model saved to: {model_path}")

        print("Validation metrics:")

        print(
            json.dumps(
                validation_metrics,
                indent=2,
            )
        )

        print("Test metrics:")

        print(
            json.dumps(
                test_metrics,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
