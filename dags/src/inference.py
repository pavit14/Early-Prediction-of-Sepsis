from typing import Dict, List, Union
from google.cloud import aiplatform, storage
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value
import csv
import io


def read_csv_instances_from_gcs(bucket_name: str, file_path: str) -> List[Dict]:
    """
    Read instances from a CSV file in Google Cloud Storage and return a list of dictionaries.
    Assumes the CSV file has a header row.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(file_path)
    content = blob.download_as_text()

    instances = []
    reader = csv.DictReader(io.StringIO(content))
    for row in reader:
        instances.append(row)
    
    return instances

def predict_custom_trained_model_from_gcs(
    project: str,
    endpoint_id: str,
    bucket_name: str,
    file_path: str,
    location: str = "us-east1",
    api_endpoint: str = "us-east1-aiplatform.googleapis.com",
):
    # Read instances from CSV file in GCS
    instances = read_csv_instances_from_gcs(bucket_name, file_path)

    # The AI Platform services require regional API endpoints.
    client_options = {"api_endpoint": api_endpoint}
    
    # Initialize client that will be used to create and send requests.
    # This client only needs to be created once, and can be reused for multiple requests.
    client = aiplatform.gapic.PredictionServiceClient(client_options=client_options)
    
    # The format of each instance should conform to the deployed model's prediction input schema.
    instances = [
        json_format.ParseDict(instance_dict, Value()) for instance_dict in instances
    ]
    
    parameters_dict = {}
    parameters = json_format.ParseDict(parameters_dict, Value())
    
    endpoint = client.endpoint_path(
        project=project, location=location, endpoint=endpoint_id
    )
    
    # Make prediction request
    response = client.predict(
        endpoint=endpoint, instances=instances, parameters=parameters
    )
    
    print("Response")
    print("Deployed Model ID:", response.deployed_model_id)
    
    # The predictions are a google.protobuf.Value representation of the model's predictions.
    predictions = response.predictions
    for prediction in predictions:
        print("Prediction:", dict(prediction))

# Example usage with GCS file
bucket_name = "sepsis_pred_bucket"
file_path = "/data/valid/valid.csv"
predict_custom_trained_model_from_gcs(
    project="497741562136",
    endpoint_id="1974151137439252480",
    location="us-east1",
    bucket_name=bucket_name,
    file_path=file_path
)