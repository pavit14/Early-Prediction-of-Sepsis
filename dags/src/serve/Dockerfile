FROM python:3.9-slim

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY serve/predict.py /app/

# Install Flask and google-cloud-storage
RUN pip install Flask google-cloud-storage joblib scikit-learn grpcio gcsfs python-dotenv tensorflow==2.15.0 numpy pandas 

ENV AIP_STORAGE_URI=gs://timeseries_data_model_bucket/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV BUCKET_NAME=timeseries_data_model_bucket
ENV PROJECT_ID=modified-link-407805


# Run serve.py when the container launches
ENTRYPOINT ["python", "predict.py"]
