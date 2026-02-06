from kfp.dsl import component, Input, Metrics, Output
from google_cloud_pipeline_components.types.artifact_types import UnmanagedContainerModel
@component(
    base_image="python:3.9",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.2", "google-cloud-bigquery", "joblib==1.3.2", "pyarrow" ,"db-dtypes","google-cloud-pipeline-components"]
)
def evaluate_model(
    project: str,
    dataset_id: str,
    feature_table_id: str,
    target_column: str,
    model_input: Input[UnmanagedContainerModel],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import bigquery
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    # 1. Load Data in batches - only sample 10000 rows to avoid memory issues
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset_id}.{feature_table_id}` LIMIT 10000"
    df = client.query(query).to_dataframe()
    
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # 2. Load Model
    model_path = os.path.join(model_input.path, "model.joblib")
    model = joblib.load(model_path)

    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    
    # 4. Log to Vertex AI UI
    metrics.log_metric("accuracy", float(acc))
    metrics.log_metric("precision", float(precision))
    metrics.log_metric("recall", float(recall))
    metrics.log_metric("f1_score", float(f1))