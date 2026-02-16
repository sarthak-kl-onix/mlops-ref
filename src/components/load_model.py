from kfp.dsl import (Output,
                        component, Artifact)
import os

project_id = os.getenv("PROJECT_ID")
region = os.getenv('REGION')

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform", "google-cloud-storage"]
)
def load_model(
    project_id: str,
    location: str,
    model_id: str,
    alias: str,
    model_output: Output[Artifact],
):
    '''
    This component loads a machine learning model from Google Cloud AI Platform Model Registry
    and saves the model artifacts to a specified output location.
    '''
    import os
    from google.cloud import aiplatform, storage

    def fetch_model_artifacts_to_local(model_uri: str, dest_dir: str):

        client = storage.Client()
        bucket_name, prefix = model_uri.replace("gs://", "").split("/", 1)
        os.makedirs(dest_dir, exist_ok=True)

        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            filename = os.path.basename(blob.name)
            if not filename:
                continue
            local_path = os.path.join(dest_dir, filename)
            try:
                blob.download_to_filename(local_path)
            except Exception as e:
                data = blob.download_as_bytes()
                with open(local_path, "wb") as f:
                    f.write(data)

    aiplatform.init(project=project_id, location=location)
    model = aiplatform.Model(
        model_name=f"projects/{project_id}/locations/{location}/models/{model_id}@{alias}"
    )
    uri = model.uri     
    fetch_model_artifacts_to_local(uri, model_output.path)
