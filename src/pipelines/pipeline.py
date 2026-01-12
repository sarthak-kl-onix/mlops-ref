import json
import os

from google_cloud_pipeline_components.v1.custom_job.utils import (
    create_custom_training_job_op_from_component
)
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.model import ModelGetOp
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
from kfp import (compiler, dsl)
from kfp.dsl import pipeline
from src.components.bq_load import bq_load
from src.components.fetch_validation_script import fetch_validation_script
from src.components.check_validation_result import check_validation_result
from src.components.prepare_features import prepare_features

env = os.getenv("ENV")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, f"../config/config-{env}.json")
with open(config_path) as json_file:
    config = json.load(json_file)

PIPELINE_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_URI = config.get("staging_bucket_uri")
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline-runs/"


@pipeline(name=PIPELINE_NAME, pipeline_root=PIPELINE_ROOT)
def pipeline_func(
    project: str,
    region: str,
    dataset_id: str,
    source_dataset_region: str,
    source_gcs_uri: str,
    source_table_id: str,
    feature_table_id: str,
    required_columns: list,
    write_disposition_bq_load: str,
    top_k_feat_prep: int,
    write_disposition_feature_load: str,
    validation_script_path: str,
    validation_sql_params: dict
) -> None:
    notify = VertexNotificationEmailOp(recipients=["sarthak.lohani@onixnet.com"])
    with dsl.ExitHandler(notify):
        bq_load_op = bq_load(
            project = project,
            region = source_dataset_region,
            gcs_source_uri = source_gcs_uri,
            dataset_id = dataset_id,
            table_id = source_table_id,
            write_disposition = write_disposition_bq_load,
        )

        get_validation_script_op = fetch_validation_script(
            validation_script_path=validation_script_path,
            sql_params = validation_sql_params
        )
        get_validation_script_op.after(bq_load_op)

        run_validation_op = BigqueryQueryJobOp(
            project=project,
            location=source_dataset_region,
            query=get_validation_script_op.outputs["output_script"],
        )   
        run_validation_op.after(get_validation_script_op)

        check_validation_result_op = check_validation_result(
            project_id=project,
            dataset_id=dataset_id,
            region = source_dataset_region
        )
        check_validation_result_op.after(run_validation_op)

        with dsl.If(check_validation_result_op.outputs['validation_result'] == True):
            # Here we are using pandas for preparing features
            # if data is too large then we can consider using Dataflow or Dataproc for this step -> DataprocPySparkBatchOp, DataprocSparkBatchOp, etc.
            prepare_features_op = prepare_features(
                project=project,
                region=region,
                dataset_id=dataset_id,
                raw_data_bq_table=source_table_id,
                feature_bq_table=feature_table_id,
                required_columns= required_columns,
                top_k= top_k_feat_prep,
                write_disposition=write_disposition_feature_load,
            )
            prepare_features_op.after(check_validation_result_op)

compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PACKAGE_PATH)
