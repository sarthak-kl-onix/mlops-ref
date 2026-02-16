from kfp.dsl import component


@component(
    base_image="python:3.10",
    packages_to_install=[
        "pandas",
        "numpy",
        "google-cloud-bigquery",
        "db-dtypes",
    ],
)
def prepare_features(
    project: str,
    region: str,
    raw_data_bq_table: str,
    feature_bq_table: str,
    required_columns: list,
    dataset_id: str,
    is_monitoring: bool,
    top_k: int = 5,
    write_disposition: str = "WRITE_TRUNCATE",
):
    """
    Preprocess raw healthcare data and write model-ready features to BigQuery.
    """

    import re
    import pandas as pd
    from google.cloud import bigquery
    from google.api_core import exceptions as gcp_exceptions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def normalize_whitespace(s):
        if pd.isna(s):
            return s
        return re.sub(r"\s+", " ", str(s)).strip()

    def normalize_blood_type(s):
        if pd.isna(s):
            return "UNKNOWN"
        x = str(s).strip().upper().replace(" ", "")
        if re.fullmatch(r"(A|B|AB|O)[+-]", x):
            return x
        return "UNKNOWN"

    def topk_map(series: pd.Series, k: int) -> pd.Series:
        series = series.fillna("UNKNOWN")
        top = series.value_counts().nlargest(k).index
        return series.apply(lambda v: v if v in top else "OTHER")

    def bucket_age(age):
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

    # ------------------------------------------------------------------
    # BigQuery Init
    # ------------------------------------------------------------------

    client = bigquery.Client(project=project, location=region)

    raw_table_id = f"{project}.{dataset_id}.{raw_data_bq_table}"
    dest_table_id = f"{project}.{dataset_id}.{feature_bq_table}"

    # ------------------------------------------------------------------
    # Read BigQuery
    # ------------------------------------------------------------------

    try:
        if is_monitoring:
            date_filter = "WHERE DATE(ingestion_time) = CURRENT_DATE()"
        else:
            date_filter = ""
        if required_columns:
            cols = ", ".join([f"`{c}`" for c in required_columns])
            query = f"SELECT {cols} FROM `{raw_table_id}` {date_filter}"
        else:
            query = f"SELECT * FROM `{raw_table_id}` {date_filter}"

        df = (
            client.query(query)
            .result()
            .to_dataframe(create_bqstorage_client=False)
        )

        print(f"Read {len(df)} rows from {raw_table_id}")

    except gcp_exceptions.GoogleAPICallError as e:
        raise RuntimeError(f"BigQuery read failed: {e}") from e

    if df.empty:
        raise RuntimeError(f"Raw table {raw_table_id} is empty")

    # ------------------------------------------------------------------
    # Rename Columns
    # ------------------------------------------------------------------

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
        "Test Results": "test_results",
        "ingestion_time": "feature_timestamp",
    }

    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})

    # ------------------------------------------------------------------
    # Deduplication
    # ------------------------------------------------------------------
    df = df.drop_duplicates(subset=["name", "age"], keep="first")
    print(f"Deduplicated data has {len(df)} rows")
    
    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].str.strip()

    df["age"] = df["age"].mask((df["age"] < 0) | (df["age"] > 120))
    df["age_bucket"] = df["age"].apply(bucket_age)
    df["age"] = (df["age"].astype(str).str.replace("–", "-", regex=False)  # EN DASH → ASCII
)
    df["blood_type"] = df["blood_type"].apply(normalize_blood_type)

    df["medical_condition"] = (
        df["medical_condition"]
        .astype(str)
        .apply(lambda s: normalize_whitespace(s).lower() if s != "nan" else "unknown")
    )

    df["hospital"] = df["hospital"].astype(str).apply(normalize_whitespace)
    df["hospital"] = topk_map(df["hospital"], top_k)

    # ------------------------------------------------------------------
    # DATE FIX (Timestamp vs date bug)
    # ------------------------------------------------------------------

    df["date_of_admission"] = pd.to_datetime(
        df["date_of_admission"], errors="coerce"
    )

    df["discharge_date"] = pd.to_datetime(
        df["discharge_date"], errors="coerce"
    )

    df["admit_time_days"] = (
        df["discharge_date"] - df["date_of_admission"]
    ).dt.days

    df["admit_time_days"] = df["admit_time_days"].mask(
        (df["admit_time_days"] < 0) | (df["admit_time_days"] > 365)
    )

    median_los = df["admit_time_days"].median(skipna=True)
    df["admit_time_days"] = df["admit_time_days"].fillna(median_los)

    df["admit_time_bucket"] = pd.cut(
        df["admit_time_days"],
        bins=[-1, 0, 2, 5, 10, 30, 365],
        labels=["0", "1–2", "3–5", "6–10", "11–30", "30+"],
    )
    df["admit_time_bucket"] = (
        df["admit_time_bucket"]
        .astype(str)
        .str.replace("–", "-", regex=False)  # EN DASH → ASCII
    )

    # ------------------------------------------------------------------
    # Create a unique record_id 
    # ------------------------------------------------------------------
    df["record_id"] = (
        df["name"].astype(str).str.lower().str.replace(" ", "") + 
        "_" + 
        df["age"].astype(str)
    )
    print("Number of records in df: ", len(df))
    # ------------------------------------------------------------------
    # Drop PII / Select Features
    # ------------------------------------------------------------------

    allowlist = [
        "record_id",        
        "age_bucket",
        "gender",
        "blood_type",
        "medical_condition",
        "admission_type",
        "hospital",
        "admit_time_bucket",
        "test_results",
        "feature_timestamp"
    ]

    features_df = df[[c for c in allowlist if c in df.columns]].copy()

    # ------------------------------------------------------------------
    # FIX: Handle Categoricals SAFELY
    # ------------------------------------------------------------------

    for c in features_df.columns:
        if pd.api.types.is_categorical_dtype(features_df[c]):
            if "UNKNOWN" not in features_df[c].cat.categories:
                features_df[c] = features_df[c].cat.add_categories(["UNKNOWN"])
            features_df[c] = features_df[c].fillna("UNKNOWN")
        else:
            features_df[c] = features_df[c].fillna("UNKNOWN").astype(str)

    # Optional hardening (safe for BigQuery ML)
    features_df = features_df.astype(str)
    print(f"Prepared features dataframe has {features_df.shape[0]}")
    # ------------------------------------------------------------------
    # Write to BigQuery
    # ------------------------------------------------------------------

    try:
        job_config = bigquery.LoadJobConfig(
            write_disposition=(
                bigquery.WriteDisposition.WRITE_APPEND
                if write_disposition == "WRITE_APPEND"
                else bigquery.WriteDisposition.WRITE_TRUNCATE
            )
        )

        load_job = client.load_table_from_dataframe(
            features_df,
            destination=dest_table_id,
            job_config=job_config,
        )

        load_job.result()

        print(
            f"Wrote {features_df.shape[0]} rows "
            f"and {features_df.shape[1]} columns to {dest_table_id}"
        )

    except gcp_exceptions.GoogleAPICallError as e:
        raise RuntimeError(f"BigQuery write failed: {e}") from e
