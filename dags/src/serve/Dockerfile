#FROM python:3.12-slim

FROM tensorflow/tensorflow:latest

# Set the working directory in the container to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY serve/predict.py /app/

# Install Flask and other dependencies
RUN pip install --upgrade pip 
RUN pip uninstall -y blinker || true
RUN pip install Flask==2.0.0 google-cloud-storage joblib scikit-learn grpcio gcsfs python-dotenv

# Uninstall blinker if it exists (ignore any errors during uninstallation)


# Set environment variables
ENV AIP_STORAGE_URI=gs://sepsis_pred_bucket/model
ENV AIP_HEALTH_ROUTE=/ping
ENV AIP_PREDICT_ROUTE=/predict
ENV AIP_HTTP_PORT=8080
ENV BUCKET_NAME=sepsis_pred_bucket
ENV PROJECT_ID=modified-link-407805

# Run the predict.py script when the container launches
ENTRYPOINT ["python", "predict.py"]
