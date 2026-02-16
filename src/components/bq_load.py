from kfp.dsl import (component, Output, Dataset)
from typing import NamedTuple

@component(
    base_image = "python:3.10",
    packages_to_install = ["google-cloud-bigquery", "google-cloud-storage"]
)
def bq_load(
    project: str,
    gcs_source_dir: str,
    dataset_id: str,
    table_id: str,
    write_disposition: str,
    raw_data: Output[Dataset],
    is_monitoring: bool,
    region: str = "US",
)-> NamedTuple("Outputs", [("load_result", bool)]):
    
    from google.cloud import bigquery
    from google.cloud import storage
    from google.api_core.exceptions import NotFound
    from datetime import datetime, timedelta, timezone

    client = bigquery.Client(project=project, location=region)
    storage_client = storage.Client(project=project)

    # 1. Clean the base URI
    if not gcs_source_dir.startswith("gs://"):
        raise ValueError("gcs_source_dir must start with 'gs://'")
    
    # Ensure base path ends with / for easy concatenation
    base_uri = gcs_source_dir if gcs_source_dir.endswith("/") else gcs_source_dir + "/"
    
    # 2. Determine target file path based on mode
    if is_monitoring:
        # Monitoring looks for: gs://.../YYYY-MM-DD/dataset.csv
        yesterday_str = (datetime.now(timezone.utc) - timedelta(days=1)).strftime('%Y-%m-%d')
        target_gcs_uri = f"{base_uri}{yesterday_str}/dataset.csv"
    else:
        # Training looks for: gs://.../training_data/healthcare_dataset.csv
        target_gcs_uri = f"{base_uri}training_data/healthcare_dataset.csv"

    print(f"Targeting URI: {target_gcs_uri}")

    # 3. Extract bucket and blob for existence check
    uri_parts = target_gcs_uri[5:].split("/", 1)
    bucket_name = uri_parts[0]
    blob_name = uri_parts[1]

    # 4. Check if the file actually exists
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    if not blob.exists():
        if is_monitoring:
            print(f" Monitoring: No data found for {yesterday_str} at {target_gcs_uri}")
            return (False,)
        else:
            raise FileNotFoundError(f" Training: Required file not found at {target_gcs_uri}")
    
    table_path = f"{project}.{dataset_id}.{table_id}"
    try:
        client.get_table(table_path)
    except NotFound:
        print(f"Table {table_path} not found. Creating with ingestion_time column...")
        # We create an empty table first to define the special column
        # BigQuery will then use 'autodetect' to add the rest of the columns during the load
        query = f"""
            CREATE TABLE `{table_path}` (
                ingestion_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
            )
        """
        client.query(query).result()

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=write_disposition,
        autodetect=True,
        # 'allow_jagged_rows' allows the CSV to have fewer columns than the BQ table
        # which lets BQ fill our 'ingestion_time' with the default value
        allow_jagged_rows=True,
        # Allow adding new columns from the CSV to the existing table schema
        schema_update_options=[
            bigquery.SchemaUpdateOption.ALLOW_FIELD_ADDITION
        ],
    )
    try:
        load_job = client.load_table_from_uri(
            target_gcs_uri,
            table_path,
            job_config=job_config,
        )

        load_job.result()  # Waits for the job to complete.
        print(f"Loaded {load_job.output_rows} rows into {dataset_id}:{table_id}.")
        
    except Exception as e:
        raise RuntimeError(
            f"BigQuery load failed for {target_gcs_uri} "
            f"into {project}.{dataset_id}.{table_id}: {e}"
        ) from e

    raw_data.uri = "bq://" + project + "." + dataset_id + "." + table_id
    return (True,)