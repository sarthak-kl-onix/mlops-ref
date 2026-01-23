from kfp.dsl import component, Output, Model, Metrics

@component(
    base_image="python:3.9",
    packages_to_install=["pandas", "scikit-learn", "google-cloud-bigquery", "pyarrow", "joblib", "db-dtypes"]
)
def train_custom_model(
    project: str,
    dataset_id: str,
    feature_table_id: str,
    target_column: str,
    model_output: Output[Model]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import bigquery
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    # 1. Load Data
    client = bigquery.Client(project=project)
    query = f"SELECT * FROM `{project}.{dataset_id}.{feature_table_id}`"
    df = client.query(query).to_dataframe()

    # 2. Split Features and Target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Define Preprocessing for Categorical Data
    categorical_features = X.columns.tolist()
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # 4. Create Training Pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # 5. Train
    pipeline.fit(X, y)

    # 6. Save Model Artifact
    os.makedirs(model_output.path, exist_ok=True)
    model_file = os.path.join(model_output.path, "model.joblib")
    joblib.dump(pipeline, model_file)