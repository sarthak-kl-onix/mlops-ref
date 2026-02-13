import json
import os
from kfp.dsl import importer

from google_cloud_pipeline_components.v1.custom_job.utils import (
    create_custom_training_job_op_from_component
)
from google_cloud_pipeline_components.types import artifact_types
from google_cloud_pipeline_components.v1.model import ModelUploadOp
from google_cloud_pipeline_components.v1.endpoint import EndpointCreateOp, ModelDeployOp
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
from vertexai.resources.preview import ml_monitoring
from google.cloud.aiplatform_v1beta1.types import ExplanationSpec, ExplanationParameters, ExplanationMetadata


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
    validation_sql_params: dict,
    model_name: str,
    model_type: str,
    target_column: str,
    data_split_method: str = "AUTO_SPLIT",
    data_split_eval_fraction: float = 0.2,
    auto_class_weights: bool = True,
    max_iterations: int = 20,
    l1_reg: float = 0.0,
    l2_reg: float = 0.0,
) -> None:
    notify = VertexNotificationEmailOp(recipients=["aditya.konde@onixnet.us"])
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



        #     # Train BQML model

        #changes in config  file for BQML model training
            # "model_name": "healthcare_test_results_classifier",
            # "model_type": "LOGISTIC_REG",
            # "target_column": "test_results",
            # "data_split_method": "AUTO_SPLIT",
            # "auto_class_weights": "TRUE",
            # "max_iterations": 20,
            # "l1_reg": 0.0,
            # "l2_reg": 0.0

        #     train_model_op = BigqueryCreateModelJobOp(
        #     project=project,
        #     location=source_dataset_region,
        #     query=f"""
        #         CREATE OR REPLACE MODEL `{project}.{dataset_id}.{model_name}`
        #         OPTIONS(
        #             model_type = '{model_type}',
        #             input_label_cols = ['{target_column}'],
        #             data_split_method = '{data_split_method}',
        #             auto_class_weights = {auto_class_weights},
        #             max_iterations = {max_iterations},
        #             l1_reg = {l1_reg},
        #             l2_reg = {l2_reg}
        #         ) AS
        #         SELECT *
        #         FROM `{project}.{dataset_id}.{feature_table_id}`
        #     """
        # )

        #     train_model_op.after(prepare_features_op)
        #     print(BigqueryExportModelJobOp.component_spec.outputs)

        # #     # Evaluate BQML model
        #     evaluate_model_op = BigqueryEvaluateModelJobOp(
        #         project=project,
        #         location=source_dataset_region,
        #         model=train_model_op.outputs["model"]
        #     )
        #     evaluate_model_op.after(train_model_op)

        #     # Export and Upload BQML model to Vertex AI Model Registry

        #     export_model_op = BigqueryExportModelJobOp(
        #         project=project,
        #         location=source_dataset_region,
        #         model=train_model_op.outputs["model"],
        #         model_destination_path=BUCKET_URI + "/models",
        #     )
        #     export_model_op.after(evaluate_model_op)

        #     unmanaged_model = importer(
        #         artifact_uri=export_model_op.outputs["exported_model_path"],
        #         artifact_class=UnmanagedContainerModel,
        #         metadata={
        #             "containerSpec": {
        #                 "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
        #             }
        #         }
        #     )
        # #     # Upload model to Vertex AI Model Registry
        #     upload_model_op = ModelUploadOp(
        #         project=project,
        #         location=region,
        #         display_name=f"{model_name}-uploaded",
        #         unmanaged_container_model=unmanaged_model.output,
        #     )
        #     upload_model_op.after(export_model_op)#won't work  





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



            # Custom Upload 
            # @dsl.component
            # def get_uri(artifact: dsl.Input[dsl.Model]) -> str:
            #     return artifact.uri
            # uri_op = get_uri(artifact=train_op.outputs['model_output'])

            # import_unmanaged_model_op = importer(
            #     artifact_uri=uri_op.output, 
            #     artifact_class=artifact_types.UnmanagedContainerModel,
            #     metadata={
            #         "containerSpec": {
            #             "imageUri": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
            #         }
            #     }
            # )
            upload_op = ModelUploadOp(
                project=project,
                location=region,
                display_name=model_name,
                unmanaged_container_model=train_op.outputs['model_output']
            )
            upload_op.after(evaluate_op)
    
            # Deploy the Model 
            endpoint_create_op = EndpointCreateOp(
                project=project,
                location=region,
                display_name=f"{model_name}-endpoint",
            )
            model_deploy_op = ModelDeployOp(
                model=upload_op.outputs["model"],
                endpoint=endpoint_create_op.outputs["endpoint"],
                dedicated_resources_machine_type="n1-standard-8",
                dedicated_resources_min_replica_count=1,
                dedicated_resources_max_replica_count=1,
                traffic_split={"0": 100}, # 100% traffic to this new model
            )
            model_deploy_op.after(upload_op)

            # Define Monitoring Schema. For AutoML models, this is optional if the schema information is available.

compiler.Compiler().compile(pipeline_func=pipeline_func, package_path=PACKAGE_PATH)
