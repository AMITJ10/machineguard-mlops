# MachineGuard MLOps

> End-to-End Predictive Maintenance MLOps project demonstrating data
> validation, feature engineering, model training, experiment tracking,
> deployment, monitoring, and cloud delivery.

## Overview

MachineGuard predicts industrial machine failures using a
production-ready MLOps pipeline.

## Features

-   End-to-end Scikit-learn pipeline
-   Pandera data validation
-   Feature engineering
-   MLflow experiment tracking
-   MLflow Model Registry
-   Quality gate
-   FastAPI inference API
-   Pydantic request validation
-   Prediction logging
-   Streamlit dashboard
-   Batch prediction
-   Docker
-   GitHub Actions CI
-   Evidently monitoring
-   Drift detection
-   Airflow retraining DAG
-   MinIO local object storage
-   Amazon S3 model storage
-   Render deployment
-   AWS-ready architecture

## Repository Structure

``` text
api/
app/
artifacts/
configs/
data/
monitoring/
scripts/
src/
tests/
```

## Tech Stack

  Area         Technology
  ------------ --------------------------------
  Language     Python 3.11
  ML           Scikit-learn
  Tracking     MLflow
  API          FastAPI
  Frontend     Streamlit
  Validation   Pandera, Pydantic
  Monitoring   Evidently, Prometheus, Grafana
  Workflow     Airflow
  Storage      MinIO, Amazon S3
  CI/CD        GitHub Actions
  Containers   Docker
  Deployment   Render

## Pipeline

1.  Data ingestion
2.  Validation
3.  Feature engineering
4.  Training
5.  Evaluation
6.  MLflow tracking
7.  Model registration
8.  Quality gate
9.  Deployment
10. Monitoring
11. Retraining

## Local Setup

``` bash
git clone <repository-url>
cd machineguard-mlops
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Train:

``` bash
python -m scripts.train
```

Run API:

``` bash
uvicorn api.main:app --reload
```

Run Streamlit:

``` bash
streamlit run app/Home.py
```

## API Endpoints

-   GET /
-   GET /health
-   GET /ready
-   GET /metrics
-   POST /predict

## Deployment

The production model is uploaded to Amazon S3. On startup the API
downloads the latest model from S3, loads it into memory, and serves
predictions. The API is deployed on Render using Docker.

## Performance

Example metrics from the latest training run:

-   Accuracy: 98.85%
-   Precision: 85.71%
-   Recall: 79.41%
-   F1-score: 82.44%
-   ROC-AUC: 0.9793

## Future Improvements

-   Automated retraining approvals
-   Canary deployments
-   Feature store integration
-   Kubernetes deployment
-   Terraform infrastructure

## Author

Amit Jadhav

------------------------------------------------------------------------

This project demonstrates the complete lifecycle of an industrial-grade
machine learning application, from data validation and model training to
cloud deployment, monitoring, and automated MLOps workflows.
