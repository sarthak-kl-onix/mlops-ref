import os
from kfp.dsl import (component, Output, Artifact)
from typing import NamedTuple

@component(
    base_image = "python:3.10",
    packages_to_install = ["google-cloud-storage"]
)
def fetch_validation_script(
    validation_script_path: str,
    sql_params: dict,
    # output_script: Output[Artifact]
) -> NamedTuple('Outputs', [('output_script', str)]):
    """
    Downloads SQL script from GCS and executes it in BigQuery.
    Returns: job_id (string) of the executed BigQuery job.
    """
    from google.cloud import storage

    if not validation_script_path.startswith("gs://"):
        raise ValueError("validation_script_path must start with gs://")

    _, path_after = validation_script_path.split("gs://", 1)
    bucket_name, _, blob_path = path_after.partition("/")

    storage_client = storage.Client()
    blob = storage_client.bucket(bucket_name).blob(blob_path)
    raw_script = blob.download_as_text()
    formatted_script =  raw_script.format(**sql_params)

    return (formatted_script,)  # Returning as a tuple to match NamedTuple output
