from flask import Flask, jsonify, request
from google.cloud import storage
import joblib
import os
import json
import pickle
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

def initialize_variables():
    """
    Initialize environment variables.
    Returns:
        tuple: The project id and bucket name.
    """
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("BUCKET_NAME")
    return project_id, bucket_name

def initialize_client_and_bucket(bucket_name):
    """
    Initialize a storage client and get a bucket object.
    Args:
        bucket_name (str): The name of the bucket.
    Returns:
        tuple: The storage client and bucket object.
    """
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    return storage_client, bucket


def load_model(bucket, bucket_name):
    """
    Fetch and load the latest model from the bucket.
    Args:
        bucket (Bucket): The bucket object.
        bucket_name (str): The name of the bucket.
    Returns:
        _BaseEstimator: The loaded model.
    """
    latest_model_blob_name = fetch_latest_model(bucket_name)
    local_model_file_name = os.path.basename(latest_model_blob_name)
    model_blob = bucket.blob(latest_model_blob_name)
    model_blob.download_to_filename(local_model_file_name)
    
    with open(local_model_file_name, 'rb') as file:
        model = pickle.load(file)
    
    return model


def fetch_latest_model(bucket_name, prefix="model/model_"):
    """Fetches the latest model file from the specified GCS bucket.
    Args:
        bucket_name (str): The name of the GCS bucket.
        prefix (str): The prefix of the model files in the bucket.
    Returns:
        str: The name of the latest model file.
    """
    # List all blobs in the bucket with the given prefix
    blobs = storage_client.list_blobs(bucket_name, prefix=prefix)

    # Extract the timestamps from the blob names and identify the blob with the latest timestamp
    blob_names = [blob.name for blob in blobs]
    if not blob_names:
        raise ValueError("No model files found in the GCS bucket.")

    latest_blob_name = sorted(blob_names, key=lambda x: x.split('_')[-1], reverse=True)[0]

    return latest_blob_name


def delete_last_5_records(group):
    return group.iloc[:-5]


def select_last_50_or_all(group):
    return group.tail(50) if len(group) >= 50 else group

def pad_trunc(data, max_len):
    new_data = []
    zero_vec = []
    for _ in range(len(data[0][0])):
        zero_vec.append(0.0)

    for sample in data:
        if len(sample) > max_len:  # truncate
            temp = sample[:max_len]
        elif len(sample) < max_len:
            temp = sample
            number_additional_elements = max_len - len(sample)
            for _ in range(number_additional_elements):
                temp.insert(0, zero_vec)  # insert zero vectors at the beginning
        else:
            temp = sample

        new_data.append(temp)

    return new_data


def create_sequence_tr_test_data(df, feature_columns, y_label='y'):
    id_list = df['id'].unique()
    x_sequence = []
    y_local = []
    ids = []
    for idx in id_list:
        n_row_idx = df[df[0] == idx].shape[0]
        if n_row_idx >= 15:
            ids.append(idx)
            x_sequence.append(df[df[0] == idx][feature_columns].iloc[:, 1:].values.tolist())
            y_local.append(df[df[0] == idx][41].iloc[:])        
    return x_sequence, y_local, ids



@app.route(os.environ['AIP_HEALTH_ROUTE'], methods=['GET'])
def health_check():
    """Health check endpoint that returns the status of the server.
    Returns:
        Response: A Flask response with status 200 and "healthy" as the body.
    """
    return {"status": "healthy"}

@app.route(os.environ['AIP_PREDICT_ROUTE'], methods=['POST'])
def predict():
    """
    Prediction route that normalizes input data, and returns model predictions.
    Returns:
        Response: A Flask response containing JSON-formatted predictions.
    """
    file = request.files['csv_file']
    
    # Read instances from the CSV file
    csv_reader = csv.DictReader(file)
    request_instances = [row for row in csv_reader]

    predict_df = predict_df.drop(predict_df.columns[[37,38]], axis=1)
    column_to_encode = 'Gender'
    encoded_columns = pd.get_dummies(predict_df[column_to_encode], prefix=column_to_encode) 
    position = predict_df.columns.get_loc(column_to_encode) 
    predict_df.drop(column_to_encode, axis=1, inplace=True) 
    predict_df = pd.concat([predict_df.iloc[:, :position], encoded_columns, predict_df.iloc[:, position:]], axis=1)
    predict_df = predict_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(delete_last_5_records)
    predict_df = predict_df.groupby(predict_df['id'], group_keys=False).apply(select_last_50_or_all)

    mean_values = predict_df.mean()
    predict_df = predict_df.fillna(mean_values)
    predict_df = predict_df.fillna(0)
    scaler = MinMaxScaler()
    scaler.fit(predict_df.iloc[:, 1:-1])
    predict_df.iloc[:, 1:-1] = scaler.transform(predict_df.iloc[:, 1:-1])
    
    feature_columns = predict_df.iloc[:, :-1].columns
    X_predict, y_predict, id_ls_predict = create_sequence_tr_test_data(predict_df, feature_columns)

    batch_size = 32
    embedding_dim = 39
    max_len = 50

    X_predict = pad_trunc(X_predict, max_len)
    y_predict = pad_sequences(y_predict, padding='pre', dtype='float32')  
    X_predict = np.reshape(X_predict, (len(X_predict), max_len, embedding_dim))
    y_predict = np.array(y_predict)

    ## import model

    predict_predict = model.predict(X_predict)
    binary_predictions_predict = (predict_predict >= 0.005).astype(int)
    binary_predictions_predict_flat = np.ravel(binary_predictions_predict)

    # Make predictions with the model
    prediction = binary_predictions_predict_flat.tolist()
    output = {'predictions': [{'result': pred} for pred in prediction]}
    return jsonify(output)

project_id, bucket_name = initialize_variables()
storage_client, bucket = initialize_client_and_bucket(bucket_name)
model = load_model(bucket, bucket_name)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
