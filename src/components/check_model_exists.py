import os
from kfp.dsl import component
from typing import NamedTuple

project_id = os.getenv("PROJECT_ID")
region = os.getenv('REGION')


@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform"]
)
def check_model_exists(model_name: str, project: str, location: str) -> NamedTuple('Outputs',[('model_id', str)]):
    from google.cloud import aiplatform
    import re

    try:
        aiplatform.init(project=project, location=location)
        filter_expression = f'displayName="{model_name}"'
        models = aiplatform.Model.list(filter=filter_expression)

        if len(models) == 0:
            return ("None",)
        else:
            m = re.search(r"models/([^/]+)$", models[0].resource_name)
            model_id = m.group(1)
            print(model_id)
            return (model_id,)
        
    except Exception:
        print(f"Model {model_name} not found")
        return ("None",)