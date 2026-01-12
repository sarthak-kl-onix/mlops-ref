from kfp.dsl import (component, Output, Input, Artifact)
from typing import NamedTuple

@component(
    base_image = "python:3.10",
    packages_to_install = ["google-cloud-bigquery"]
)
def check_validation_result(
    project_id: str,
    dataset_id: str, 
    result_output: Output[Artifact],
    region: str = "US", 
) -> NamedTuple('Outputs', [('validation_result', bool)]):
    import json, os
    from google.cloud import bigquery

    os.makedirs(result_output.path, exist_ok=True)
    output_dir = os.path.join(result_output.path + '/validation_result.json')

    client = bigquery.Client(project=project_id, location=region)
    query = f"""
     SELECT * from 
     `{project_id}.{dataset_id}.validation_results`
     where validated_table = '{project_id}.{dataset_id}.healthcare_raw_data'
     order by run_ts desc
     limit 1;
    """
    try:
        query_job = client.query(query)
        results = query_job.result()
        row = dict(next(results, None))
        if row: 
            with open(output_dir, 'w') as f:
                json.dump(row, f, default=str)
            return (row['passed'],)
    except Exception as e:
        raise RuntimeError(f"Failed to execute query")
    
