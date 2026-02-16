from kfp.dsl import (component, Input, Output, Dataset)
from google_cloud_pipeline_components.types.artifact_types import UnmanagedContainerModel

@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "google-cloud-bigquery", "joblib==1.3.2", "pyarrow", "db-dtypes", "google-cloud-pipeline-components"]
)
def calculate_metrics(
    project: str,
    region: str,
    dataset_id: str,
    feature_table_id: str,
    target_column: str,
    model_input: Input[UnmanagedContainerModel],
    metrics_history_table: str,
    metrics_output: Output[Dataset]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import bigquery
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from datetime import datetime

    client = bigquery.Client(project=project, location=region)

    # 1. Fetch Today's Features (those created in the previous step)
    # We filter by CURRENT_DATE() to match our bq_load logic
    query = f"""
        SELECT * FROM `{project}.{dataset_id}.{feature_table_id}` 
        WHERE DATE(feature_timestamp) = CURRENT_DATE()
    """
    features_df = client.query(query).to_dataframe()
    
    if features_df.empty:
        print("No new features found for today. Skipping metrics calculation.")
        return

    # 2. Load the Model and Predict
    model_dir = model_input.path.rstrip("/")
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    
    # Drop metadata columns before prediction
    X = features_df.drop(columns=[target_column, "record_id", "feature_timestamp"], errors='ignore')
    y_true = features_df[target_column]

    # 4. Generate Predictions
    y_pred = model.predict(X)

    # 5. Calculate Classification Metrics
    # We use 'weighted' average to handle potential class imbalance in healthcare data
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # 6. Prepare Results for History Table
    results = {
        "run_ts": datetime.now(),
        "model_name": model_input.metadata.get("displayName", "unknown_model"),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "sample_size": len(features_df)
    }

    # 7. Append to BigQuery Monitoring Table
    # This acts as your "Performance Ledger"
    metrics_df = pd.DataFrame([results])
    table_ref = f"{project}.{dataset_id}.{metrics_history_table}"
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    try:
        client.load_table_from_dataframe(metrics_df, table_ref, job_config=job_config).result()
        print(f"Monitoring metrics logged to {table_ref}")
    except Exception as e:
        print(f"Failed to log metrics to BigQuery: {e}")
        raise RuntimeError(f"Failed to log metrics to BigQuery: {e}")

    # Output print for logs
    print(f"Today's Performance -> F1: {f1:.4f}, Accuracy: {acc:.4f}")
    metrics_output.uri = f"bq://{table_ref}"