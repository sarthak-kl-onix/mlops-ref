import json
import os

from google_cloud_pipeline_components.v1.custom_job.utils import (
    create_custom_training_job_op_from_component
)
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.model import ModelGetOp
from google_cloud_pipeline_components.v1.bigquery import (
    BigqueryQueryJobOp,
    BigqueryCreateModelJobOp,
    BigqueryEvaluateModelJobOp,
    BigqueryExportModelJobOp
)
from google_cloud_pipeline_components.v1.vertex_notification_email import VertexNotificationEmailOp
from kfp import (compiler, dsl)
from kfp.dsl import pipeline
from src.components.bq_load import bq_load
from src.components.fetch_validation_script import fetch_validation_script
from src.components.check_validation_result import check_validation_result
from src.components.prepare_features import prepare_features
from src.components.train_model import train_custom_model
from src.components.evaluate_model import evaluate_model
from src.components.check_model_exists import check_model_exists


env = os.getenv("ENV")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, f"../../config/config-{env}.json")
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
    dataset_region: str,
    source_gcs_dir: str,
    source_table_id: str,
    feature_table_id: str,
    required_columns: list,
    write_disposition_bq_load: str,
    top_k_feat_prep: int,
    write_disposition_feature_load: str,
    validation_script_path: str,
    validation_sql_params: dict,
    model_name: str,
    model_labels: dict,
    version_aliases: list,
    model_type: str,
    target_column: str,
    data_split_method: str = "AUTO_SPLIT",
    data_split_eval_fraction: float = 0.2,
    auto_class_weights: bool = True,
    max_iterations: int = 20,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> None:
    notify = VertexNotificationEmailOp(recipients=["sarthak.lohani@onixnet.com"])
    with dsl.ExitHandler(notify):
        bq_load_op = bq_load(
            project = project,
            region = dataset_region,
            gcs_source_dir = source_gcs_dir,
            dataset_id = dataset_id,
            table_id = source_table_id,
            write_disposition = write_disposition_bq_load,
            is_monitoring = False
        )

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
                is_monitoring=False,
                write_disposition=write_disposition_feature_load,
            )
            prepare_features_op.after(check_validation_result_op)

            #changes in config file for custom model training
            # "model_name": "healthcare_test_results_classifier2",
            # "model_type": "SKLEARN_RANDOM_FOREST",
            # "target_column": "test_results", 
            # "data_split_method": "AUTO_SPLIT",
            # "auto_class_weights": true,
            # "max_iterations": 100,
            # "l1_reg": 0.0,
            # "l2_reg": 0.0
            #Custom Training
            train_op = train_custom_model(
                project=project,
                dataset_id=dataset_id,
                feature_table_id=feature_table_id,
                target_column=target_column
            )
            train_op.after(prepare_features_op)

            # Custom Evaluation
            evaluate_op = evaluate_model(
                project=project,
                dataset_id=dataset_id,
                feature_table_id=feature_table_id,
                target_column=target_column,
                model_input=train_op.outputs['model_output']
            )
            evaluate_op.after(train_op)

            check_model_exists_op = check_model_exists(
                model_name=model_name,
                project=project,
                location=region
            )
            check_model_exists_op.after(train_op)

            with dsl.If(check_model_exists_op.outputs["model_id"] == "None"):
                model_upload_op = ModelUploadOp(
                    project=project,
                    location=region,
                    display_name=model_name,
                    unmanaged_container_model=train_op.outputs["model_output"],
                    version_aliases=version_aliases,
                    labels=model_labels,
                )
                model_upload_op.after(check_model_exists_op)
                
            with dsl.Else():
                model_get_op = ModelGetOp(
                    project=project,
                    location=region,
                    model_name=check_model_exists_op.outputs["model_id"],
                )
                model_get_op.after(check_model_exists_op)
                
                model_upload_op = ModelUploadOp(
                    project=project,
                    location=region,
                    display_name=model_name,
                    unmanaged_container_model=train_op.outputs["model_output"],
                    parent_model = model_get_op.outputs['model'],
                    version_aliases=version_aliases,
                    labels=model_labels,
                )
                model_upload_op.after(model_get_op)
            # Define Monitoring Schema. For AutoML models, this is optional if the schema information is available.

compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PACKAGE_PATH)
