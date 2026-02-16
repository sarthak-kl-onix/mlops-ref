from kfp.dsl import pipeline
from kfp import dsl
from kfp import compiler
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
from src.components.bq_load import bq_load
from src.components.fetch_validation_script import fetch_validation_script
from src.components.check_validation_result import check_validation_result
from src.components.prepare_features import prepare_features
from src.components.check_model_exists import check_model_exists
from src.components.load_model import load_model
from src.components.calculate_metrics import calculate_metrics
from google_cloud_pipeline_components.v1.bigquery import BigqueryQueryJobOp
import json, os


env = os.getenv("ENV")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, f"../../config/config-{env}.json")
with open(config_path) as json_file:
    config = json.load(json_file)

BUCKET_URI = config.get("staging_bucket_uri")
PIPELINE_ROOT = "gs://{}/monitoring_pipeline/".format(BUCKET_URI)
PIPELINE_NAME = config.get('pipeline_name')
PACKAGE_PATH = config.get('pipeline_package_path')

@pipeline(
    pipeline_root= PIPELINE_ROOT,
    name= PIPELINE_NAME
)
def pipeline(
    project: str,
    region: str,
    dataset_region: str,
    source_gcs_dir: str,
    dataset_id: str,
    source_table_id: str,
    write_disposition_bq_load: str,
    alias: str,
    model_name: str,
    validation_script_path: str,
    validation_sql_params: dict,
    feature_table_id: str,
    required_columns: list,
    top_k_feat_prep: int,
    target_column: str,
    monitoring_metrics_history_table: str,
    write_disposition_feature_load: str,
):
    notify = VertexNotificationEmailOp(recipients=["sarthak.lohani@onixnet.com"])
    with dsl.ExitHandler(notify):
        bq_load_op = bq_load(
            project = project,
            region = dataset_region,
            gcs_source_dir = source_gcs_dir,
            dataset_id = dataset_id,
            table_id = source_table_id,
            write_disposition = write_disposition_bq_load,
            is_monitoring = True
        )
        with dsl.If(bq_load_op.outputs['load_result'] == True):
            get_validation_script_op = fetch_validation_script(
                validation_script_path=validation_script_path,
                sql_params = validation_sql_params
            )
            get_validation_script_op.after(bq_load_op)

            run_validation_op = BigqueryQueryJobOp(
                project=project,
                location=dataset_region,
                query=get_validation_script_op.outputs["output_script"],
            )   
            run_validation_op.after(get_validation_script_op)

            check_validation_result_op = check_validation_result(
                project_id=project,
                dataset_id=dataset_id,
                region = dataset_region
            )
            check_validation_result_op.after(run_validation_op)

            with dsl.If(check_validation_result_op.outputs['validation_result'] == True):
                # Here we are using pandas for preparing features
                # if data is too large then we can consider using Dataflow or Dataproc for this step -> DataprocPySparkBatchOp, DataprocSparkBatchOp, etc.
                prepare_features_op = prepare_features(
                    project=project,
                    region=dataset_region,
                    dataset_id=dataset_id,
                    raw_data_bq_table=source_table_id,
                    feature_bq_table=feature_table_id,
                    required_columns= required_columns,
                    top_k= top_k_feat_prep,
                    is_monitoring = True,
                    write_disposition=write_disposition_feature_load,
                )
                prepare_features_op.after(check_validation_result_op)
            
                check_model_exists_op = check_model_exists(
                    project=project,
                    location=region,
                    model_name=model_name,
                )
                check_model_exists_op.after(prepare_features_op)

                with dsl.If(check_model_exists_op.outputs["model_id"] != ""):
                    load_model_op = load_model(
                        project_id = project,
                        location = region,
                        model_id = check_model_exists_op.outputs['model_id'],
                        alias = alias
                    )
                    load_model_op.after(check_model_exists_op)

                    calculate_metrics_op = calculate_metrics(
                        project=project,
                        region=dataset_region,
                        dataset_id=dataset_id,
                        feature_table_id=feature_table_id,
                        target_column=target_column,
                        model_input=load_model_op.outputs['model_output'],
                        metrics_history_table=monitoring_metrics_history_table
                    )
                    calculate_metrics_op.after(load_model_op)


compiler.Compiler().compile(
    pipeline_func=pipeline
    , package_path=PACKAGE_PATH
)


    