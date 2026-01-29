from kfp.dsl import component, Input, Model, Metrics, Output
@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-bigquery", "joblib", "pyarrow" ,"db-dtypes"]
)
def evaluate_model(
    project: str,
    dataset_id: str,
    feature_table_id: str,
    target_column: str,
    model_input: Input[Model],
    metrics: Output[Metrics]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import bigquery
    from sklearn.metrics import accuracy_score, classification_report

    # 1. Load Data (In a real scenario, use a held-out test set)
    client = bigquery.Client(project=project)
    df = client.query(f"SELECT * FROM `{project}.{dataset_id}.{feature_table_id}`").to_dataframe()
    
    X_test = df.drop(columns=[target_column])
    y_test = df[target_column]

    # 2. Load Model
    model_path = os.path.join(model_input.path, "model.joblib")
    model = joblib.load(model_path)

    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    # 4. Log to Vertex AI UI
    metrics.log_metric("accuracy", float(acc))