from kfp.dsl import (component, Output, Dataset)

@component(
    base_image = "python:3.10",
    packages_to_install = ["google-cloud-bigquery"]
)
def bq_load(
    project: str,
    gcs_source_uri: str,
    dataset_id: str,
    table_id: str,
    write_disposition: str,
    raw_data: Output[Dataset],
    region: str = "US",
) -> None:
    from google.cloud import bigquery

    client = bigquery.Client(project=project, location=region)

    # Extract the bucket name and blob name from the GCS URI
    if not gcs_source_uri.startswith("gs://"):
        raise ValueError("gcs_source_uri must start with 'gs://'")
    
    uri_parts = gcs_source_uri[5:].split("/", 1)
    bucket_name = uri_parts[0]
    blob_name = uri_parts[1] if len(uri_parts) > 1 else ""

    # Create a GCS URI for BigQuery load job
    gcs_uri = f"gs://{bucket_name}/{blob_name}"

    print(f"Loading data from {gcs_uri} into {project}.{dataset_id}.{table_id}")
    
    table_ref = client.dataset(dataset_id).table(table_id)

    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        write_disposition=write_disposition,
        autodetect=True,
    )
    try:
        load_job = client.load_table_from_uri(
            gcs_uri,
            table_ref,
            job_config=job_config,
        )

        load_job.result()  # Waits for the job to complete.
        print(f"Loaded {load_job.output_rows} rows into {dataset_id}:{table_id}.")
    except Exception as e:
        raise RuntimeError(
            f"BigQuery load failed for {gcs_source_uri} "
            f"into {project}.{dataset_id}.{table_id}: {e}"
        ) from e

    raw_data.uri = "bq://" + project + "." + dataset_id + "." + table_id