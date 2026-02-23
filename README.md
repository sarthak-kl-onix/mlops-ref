# MLOps Template Guide

This guide provides a comprehensive overview and templates for building MLOps pipelines on Google Cloud Vertex AI, based on the `mlops-ref` architecture.

## 1. Overview

The architecture consists of two main pipelines:
1.  **Training Pipeline**: Ingests data, validates it, prepares features, trains a model (Custom or AutoML/BQML), evaluates it, and registers it to the Vertex AI Model Registry.
2.  **Monitoring Pipeline**: Runs regularly to validate new data, detect drift, and monitor model performance in production.

## 2. Prerequisites

-   **Google Cloud Project** with Vertex AI API enabled.
-   **Google Cloud Storage (GCS)** bucket for staging pipeline artifacts and data.
-   **BigQuery** dataset for storing raw data, features, and logs.
-   **Service Account** with permissions to access GCS, BigQuery, and Vertex AI.

## 3. Configuration

Configuration is managed via JSON files (e.g., `src/config/config-dev.json`). This allows separate settings for dev, staging, and prod environments.

### Template: `config-dev.json`

```json
{
    "project_id": "your-gcp-project-id",
    "region": "us-central1",
    "pipeline_name": "healthcare-training-pipeline",
    "pipeline_package_path": "training_pipeline.json",
    "staging_bucket_uri": "gs://your-staging-bucket",
    "dataset_id": "healthcare_dataset",
    "source_table_id": "raw_data",
    "feature_table_id": "features",
    "model_name": "healthcare_model",
    "model_type": "SKLEARN_RANDOM_FOREST",
    "target_column": "readmitted",
    "required_columns": ["age", "diagnosis", "readmitted"]
}
```

## 4. Pipeline Templates

### 4.1 Training Pipeline (`src/pipelines/training/pipeline.py`)

Key steps:
1.  **Data Ingestion (`bq_load`)**: Loads data from GCS to BigQuery.
2.  **Data Validation (`run_validation_op`)**: Executes SQL checks (e.g., null checks) on the raw data.
3.  **Feature Preparation (`prepare_features`)**: Cleans data and creates features in BigQuery.
4.  **Training (`train_custom_model`)**: Trains a model using a custom Python component.
5.  **Evaluation (`evaluate_model`)**: Calculates metrics (accuracy, F1, etc.) on a test split.
6.  **Model Registry (`ModelUploadOp`)**: Uploads the trained model to Vertex AI Model Registry if it passes checks.

### 4.2 Monitoring Pipeline (`src/pipelines/monitoring/pipeline.py`)

Key steps:
1.  **Batch Ingestion**: Loads the latest batch of data.
2.  **Validation**: Validates the new batch.
3.  **Feature Prep**: Prepares features for the new batch.
4.  **Model Existence Check**: Verifies if a deployed model exists.
5.  **Metric Calculation**: Generates predictions using the deployed model and compares them with ground truth (if available) to calculate performance metrics.

## 5. Component Guide

Components are standalone Python functions decorated with `@component`. They are the building blocks of the pipeline.

### `bq_load`
-   **Purpose**: Loads CSV data from GCS into a BigQuery table.
-   **Key Logic**: Handles schema autodetection and adds an `ingestion_time` timestamp.
-   **Customization**: Modify `schema_update_options` if you need to enforce strict schemas.

### `prepare_features`
-   **Purpose**:Feature engineering and cleaning.
-   **Key Logic**:
    -   Reads from BigQuery.
    -   Performs pandas-based transformations (e.g., bucketizing age, normalizing strings).
    -   Writes processed features back to valid BigQuery table.
    -   **Critical**: Sets `model_output.metadata["containerSpec"]` to point to a pre-built Vertex AI serving container (e.g., `us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest`).
-   **Customization**: This is where most domain-specific logic resides. Update the `col_map` and transformation functions for your specific dataset.

### `train_custom_model`
-   **Purpose**: Trains a Scikit-learn model.
-   **Key Logic**:
    -   Reads features from BigQuery.
    -   Splits data into train/test.
    -   Trains a `statsmodel` or `sklearn` pipeline.
    -   Saves the model as `model.joblib`.
    -   **Critical**: Sets `model_output.metadata["containerSpec"]` to point to a pre-built Vertex AI serving container (e.g., `us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest`).

## 6. Usage Instructions

1.  **Setup Environment**:
    ```bash
    export ENV="dev"
    ```

2.  **Compile and Run Training Pipeline**:
    ```bash
    # From the root directory
    uv run -m src.pipelines.training.pipeline
    ```
    *Note: Ensure your `TO_RUN` script or CI/CD pipeline executes the compiled JSON or Python script.*

3.  **Compile and Run Monitoring Pipeline**:
    ```bash
    uv run -m src.pipelines.monitoring.pipeline
    ```

## 7. Best Practices

-   **Version Control**: Always tag your model versions in the registry (`version_aliases`).
-   **Experiment Tracking**: Use `Vertex AI Experiments` (integrated in the pipeline runs) to track params and metrics.
-   **Containerization**: For complex dependencies, build a custom base image instead of installing packages at runtime in the component decorator.
