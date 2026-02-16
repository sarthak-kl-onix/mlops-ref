import json
import os

from google.cloud import storage

env = os.getenv("ENV")
config_path = f"./src/config/config-{env}.json"
with open(config_path) as json_file:
    config = json.load(json_file)

print(f"config: {config}")

BUCKET_NAME = config.get("staging_bucket_uri")
BUCKET_NAME = BUCKET_NAME.replace("gs://", "")
SQL_PATH_1 = config.get("validation_script_path")
SQL_PATH_1 = SQL_PATH_1.replace(f"gs://{BUCKET_NAME}/", "")

print(f"Bucket Name: {BUCKET_NAME}, SQL_PATH_1: {SQL_PATH_1}")


def upload_files() -> None:
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    local_path = os.path.join(
        curr_dir, "../src/sql/validation_sql.sql"
    )
    print(f"local path 1: {local_path}")

    blob1 = bucket.blob(SQL_PATH_1)
    blob1.upload_from_filename(local_path)
    print(f"Uploaded: {local_path} -> gs://{BUCKET_NAME}/{SQL_PATH_1}")


if __name__ == "__main__":
    upload_files()
