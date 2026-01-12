from kfp.dsl import (component)


@component(
    base_image="python:3.10",
    packages_to_install=["pandas", "google-cloud-bigquery", "numpy"]
)
def prepare_features(
    project: str,
    region: str,
    raw_data_bq_table: str,
    feature_bq_table: str,
    required_columns: list,
    dataset_id: str,
    top_k: int = 5,
    write_disposition: str = "WRITE_TRUNCATE",
):
    '''
    We are doing preprocessing/feature engineering in this component and save the processed data to another BigQuery table.
    '''
    import re
    from google.cloud import bigquery
    import pandas as pd
    from google.api_core import exceptions as gcp_exceptions

    # --- helpers ---
    def normalize_whitespace(s: str) -> str:
        if pd.isna(s):
            return s
        return re.sub(r"\s+", " ", str(s)).strip()

    def normalize_blood_type(s: str) -> str:
        if pd.isna(s):
            return "UNKNOWN"
        x = str(s).strip().upper().replace(" ", "")
        # Allow variants like "A+", "A-"
        if re.fullmatch(r"(A|B|AB|O)[+-]", x):
            return x
        return "UNKNOWN"

    def topk_map(series: pd.Series, k: int) -> pd.Series:
        top = series.fillna("UNKNOWN").value_counts().nlargest(k).index
        return series.fillna("UNKNOWN").apply(lambda v: v if v in top else "OTHER")

    def bucket_age(age: float) -> str:
        try:
            a = float(age)
        except Exception:
            return "unknown"
        if a < 0:
            return "unknown"
        if a <= 17:
            return "0-17"
        if a <= 34:
            return "18-34"
        if a <= 54:
            return "35-54"
        if a <= 74:
            return "55-74"
        return "75+"

    # --- initialization ---
    client = bigquery.Client(project=project, location=region)

    # Build full raw table id
    raw_table_id = f"{project}.{dataset_id}.{raw_data_bq_table}"

    # Normalize destination table
    dest_table_id = f"{project}.{dataset_id}.{feature_bq_table.strip()}"

    try:
        # 1) Read raw table (optionally selecting only required columns)
        if required_columns:
            # sanitize column names for query by wrapping in backticks
            select_cols = ", ".join([f"`{c}`" for c in required_columns])
            query = f"SELECT {select_cols} FROM `{raw_table_id}`"
        else:
            query = f"SELECT * FROM `{raw_table_id}`"
        df = client.query(query).result().to_dataframe(create_bqstorage_client=False)
        print(f"Read {len(df)} rows from {raw_table_id}")
    except gcp_exceptions.GoogleAPICallError as e:
        raise RuntimeError(f"BigQuery API error when reading raw table {raw_table_id}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error reading raw table {raw_table_id}: {e}") from e

    # If empty, create an empty DataFrame with columns and exit gracefully (or raise depending policy)
    if df.shape[0] == 0:
        raise RuntimeError(f"Raw data table {raw_table_id} is empty. Cannot prepare features on empty dataset.")

    # --- Standardize column names (map to snake_case internally) ---
    col_map = {
        "Name": "name",
        "Age": "age",
        "Gender": "gender",
        "Blood Type": "blood_type",
        "Medical Condition": "medical_condition",
        "Date of Admission": "date_of_admission",
        "Doctor": "doctor",
        "Hospital": "hospital",
        "Insurance Provider": "insurance_provider",
        "Billing Amount": "billing_amount",
        "Room Number": "room_number",
        "Admission Type": "admission_type",
        "Discharge Date": "discharge_date",
        "Medication": "medication",
        "Test Results": "test_results"
    }
    rename_map = {old: new for old, new in col_map.items() if old in df.columns}
    df = df.rename(columns=rename_map)
    # --- Apply cleaning & mappings ---
    # Trim whitespace for string columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()

    # clip unrealistic values
    df["age"] = df["age"].mask((df["age"] < 0) | (df["age"] > 120))
 
    df["age_bucket"] = df["age"].apply(bucket_age)

    df["blood_type"] = df["blood_type"].apply(normalize_blood_type)

    df["medical_condition"] = df["medical_condition"].astype(str).apply(lambda s: normalize_whitespace(s).lower() if s and s != "nan" else "unknown")

    df["hospital"] = df["hospital"].astype(str).apply(normalize_whitespace)
    df["hospital"] = topk_map(df["hospital"], top_k)


    df["date_of_admission"] = pd.to_datetime(df["date_of_admission"], errors="coerce").dt.date

    df["discharge_date"] = pd.to_datetime(df["discharge_date"], errors="coerce")

    df["admit_time_days"] = (df["discharge_date"] - df["date_of_admission"]).dt.days

    # remove negative or absurd values
    df["admit_time_days"] = df["admit_time_days"].mask(
        (df["admit_time_days"] < 0) | (df["admit_time_days"] > 365)
    )
    # impute missing LOS with median
    median_los = df["admit_time_days"].median(skipna=True)
    df["admit_time_days"] = df["admit_time_days"].fillna(median_los)

    df["admit_time_bucket"] = pd.cut(
        df["admit_time_days"],
        bins=[-1, 0, 2, 5, 10, 30, 365],
        labels=["0", "1–2", "3–5", "6–10", "11–30", "30+"]
    )

    # 8) Test results label normalization
    if "test_results" in df.columns:
        def normalize_label(v):
            if pd.isna(v):
                return "Unknown"
            x = str(v).strip().lower()
            if x.startswith("norm"):
                return "Normal"
            if x.startswith("abn"):
                return "Abnormal"
            if x.startswith("inc"):
                return "Inconclusive"
            return "Other"
        df["test_results"] = df["test_results"].apply(normalize_label)
    else:
        df["test_results"] = "Unknown"

    # 9) Duplicate detection (optional) - detect duplicates by (name + date_of_admission) before dropping PII
    duplicates_detected = 0
    if ("name" in df.columns) and ("date_of_admission" in df.columns):
        dup_series = df.duplicated(subset=["name", "date_of_admission"], keep=False)
        duplicates_detected = int(dup_series.sum())
        if duplicates_detected > 0:
            print(f"Warning: {duplicates_detected} duplicate rows found by (name + date_of_admission).")
        # You may choose to drop exact duplicates:
        df = df.drop_duplicates(subset=["name", "date_of_admission"], keep="first")

    # 10) Drop PII and unsafe columns (explicit allowlist approach)
    # Allowlist of model-safe columns
    allowlist = [
        "age_bucket", "gender", "blood_type",
        "medical_condition", "admission_type",
        "hospital","admit_time_bucket",
        "test_results"
    ]
    # Keep only columns in allowlist if they exist
    final_cols = [c for c in allowlist if c in df.columns]
    features_df = df[final_cols].copy()

    # 11) Final sanity fixes: ensure no NaN in categorical columns (replace with "UNKNOWN" or "OTHER")
    cat_cols = ["gender", "blood_type", "admission_type", "hospital", "admit_time_bucket", "test_results", "age_bucket"]
    for c in cat_cols:
        if c in features_df.columns:
            features_df[c] = features_df[c].fillna("UNKNOWN").astype(str)

    # 12) Write to BigQuery (overwrite by default)
    try:
        job_config = bigquery.LoadJobConfig()
        if write_disposition == "WRITE_TRUNCATE":
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE
        elif write_disposition == "WRITE_APPEND":
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
        else:
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_TRUNCATE

        # Convert any pandas boolean to native types BQ accepts
        # BigQuery client will infer schema from DataFrame
        load_job = client.load_table_from_dataframe(
            features_df,
            destination=dest_table_id,
            job_config=job_config
        )
        load_job.result()
        print(f"Wrote {features_df.shape[0]} rows and {features_df.shape[1]} columns to {dest_table_id}")
    except gcp_exceptions.GoogleAPICallError as e:
        raise RuntimeError(f"BigQuery API error when writing features to {dest_table_id}: {e}") from e
    except Exception as e:
        raise RuntimeError(f"Error writing features to {dest_table_id}: {e}") from e