FROM python:3.12-slim

WORKDIR /

ENV AIP_STORAGE_URI=gs://sepsis_pred_bucket/model
ENV BUCKET_NAME=sepsis_pred_bucket

COPY trainer /trainer

# Install build dependencies
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pandas google-cloud-storage scikit-learn joblib grpcio gcsfs python-dotenv


ENTRYPOINT ["python", "-m", "trainer.train"]
