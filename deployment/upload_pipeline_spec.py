import requests
from google.auth import default
from google.auth.transport.requests import Request

PROJECT_ID = "search-ahmed"
REPO_ID = "ml-pipelines"
FILE_PATH = "src/pipelines/training/mlops-ref-pipeline.yaml"

# 1. Get Application Default Credentials (ADC)
#    Works if you ran: gcloud auth application-default login
credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])

# 2. Refresh token to get a valid access token
credentials.refresh(Request())
token = credentials.token

# 3. Prepare headers
headers = {
    "Authorization": f"Bearer {token}",
}

# 4. Multipart form-data body (same as curl -F)
files = {
    "content": ("mlops-ref-pipeline.yaml", open(FILE_PATH)),
}
data = {
    "tags": "v1,latest",
}

# 5. The Artifact Registry KFP Repository URL
url = f"https://us-central1-kfp.pkg.dev/{PROJECT_ID}/{REPO_ID}"

# 6. Upload the pipeline spec
response = requests.post(url, headers=headers, files=files, data=data)

print("Status:", response.status_code)
print("Response:", response.text)
