import datetime
import json
import os

from google.cloud import aiplatform

env = os.getenv("ENV")
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, f"../src/config/config-{env}.json")

with open(config_path) as json_file:
    config = json.load(json_file)

SERVICE_ACCOUNT = config.get("service_account")
DISPLAY_NAME = config.get("pipeline_name")
PACKAGE_PATH = config.get("pipeline_package_path")
BUCKET_URI = config.get("staging_bucket_uri")
PIPELINE_ROOT = f"{BUCKET_URI}/pipeline-runs/"

sql_params = {
    "PROJECT_ID": config.get("project_id"),
    "DATASET_ID": config.get("dataset_id"),
    "TABLE_ID": config.get("source_table_id"),
    "REQUIRED_COLUMNS": ", ".join(
        f"'{c}'" for c in config.get("required_columns")
    ),
    "TARGET_DATE": config.get("validation_target_date"),
    "MAX_BLOOD_TYPE": config.get("max_blood_type"),
    "MAX_NULL_ADMISSION": config.get("max_null_admission"),
    "MAX_NULL_MEDCOND": config.get("max_null_medcond")
}
parameters = {
    "project": config.get("project_id"),
    "region": config.get("region"),
    "dataset_id": config.get("dataset_id"),
    "dataset_region": config.get("dataset_region"),
    "source_gcs_dir": config.get("source_gcs_dir"),
    "source_table_id": config.get("source_table_id"),
    "feature_table_id": config.get("feature_table_id"),
    "required_columns": config.get("required_columns"),
    "top_k_feat_prep": config.get("top_k_feat_prep"),
    "write_disposition_feature_load": config.get("write_disposition_feature_load"),
    "write_disposition_bq_load": config.get("write_disposition_bq_load"),
    "validation_script_path": config.get("validation_script_path"),
    "validation_sql_params": sql_params,
    "model_name": config.get("model_name"),
    "version_aliases": config.get("version_aliases"),
    "model_labels": dict(config.get("model_labels")),
    "model_type": config.get("model_type"),
    "target_column": config.get("target_column"),
    "data_split_method": config.get("data_split_method", "AUTO_SPLIT"),
    "auto_class_weights": config.get("auto_class_weights", True),
    "max_iterations": config.get("max_iterations", 20),
    "l1_reg": config.get("l1_reg", 0.0),
    "l2_reg": config.get("l2_reg", 0.0)
}

if os.getenv('COMMIT_SHA'):
    job_id = DISPLAY_NAME + "-" + os.getenv('COMMIT_SHA')
    PACKAGE_PATH = os.getenv('PACKAGE_PATH')
else:
    job_id = DISPLAY_NAME + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S")

job = aiplatform.PipelineJob(
    display_name=DISPLAY_NAME,
    template_path=PACKAGE_PATH,
    pipeline_root=PIPELINE_ROOT,
    parameter_values=parameters,
    job_id=job_id,
    # enable_caching=False,
    project=config.get("project_id"),
    location=config.get("region")
)

print(f"deploying pipeline {DISPLAY_NAME} from {PACKAGE_PATH}")
job.submit(service_account=SERVICE_ACCOUNT)
