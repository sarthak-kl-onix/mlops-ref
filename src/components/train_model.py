from kfp.dsl import component, Output, Metrics
# Import the specialized artifact type
from google_cloud_pipeline_components.types.artifact_types import UnmanagedContainerModel

@component(
    base_image="python:3.9",
    packages_to_install=[
        "pandas==2.0.3", 
        "scikit-learn==1.3.2", 
        "google-cloud-bigquery", 
        "pyarrow", 
        "joblib==1.3.2", 
        "db-dtypes",
        "google-cloud-pipeline-components" # Required for the artifact type definition
    ]
)
def train_custom_model(
    project: str,
    dataset_id: str,
    feature_table_id: str,
    target_column: str,
    # Change the output type here
    model_output: Output[UnmanagedContainerModel]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import bigquery
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.model_selection import train_test_split

    # 1. Load Data
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset_id}.{feature_table_id}`"
    df = client.query(query).to_dataframe()

    train_df, _ = train_test_split(df, test_size=0.2, random_state=42)

    # 3. Separate Features and Target + Drop Non-Features
    # CRITICAL: We must remove record_id and feature_timestamp before training!
    drop_cols = [target_column, "record_id", "feature_timestamp"]
    X = train_df.drop(columns=[c for c in drop_cols if c in train_df.columns])
    y = train_df[target_column]

    # 3. Preprocessing
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore', max_categories=10, sparse_output=False), categorical_features)
            ],
            remainder='passthrough'
        )
    else:
        preprocessor = ColumnTransformer(transformers=[], remainder='passthrough')

    # 4. Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=30, 
            max_depth=10,
            random_state=42
        ))
    ])

    # 5. Train
    pipeline.fit(X, y)

    # 6. Save Model Artifact
    # Note: Vertex AI Sklearn containers expect the file to be named 'model.joblib'
    os.makedirs(model_output.path, exist_ok=True)
    model_file = os.path.join(model_output.path, "model.joblib")
    joblib.dump(pipeline, model_file)

    # 7. Metadata injection for ModelUploadOp
    # This is the crucial step that replaces the 'importer'
    model_output.metadata["containerSpec"] = {
        "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
    }