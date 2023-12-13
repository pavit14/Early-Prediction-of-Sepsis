#FROM python:3.12-slim


FROM tensorflow/tensorflow:latest

WORKDIR /

ENV AIP_STORAGE_URI=gs://sepsis_pred_bucket/model
ENV BUCKET_NAME=sepsis_pred_bucket

COPY trainer /trainer

# Install any additional dependencies or packages if needed
# RUN apt-get update \
#     && apt-get install -y build-essential \
#     && apt-get clean \
#     && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "-m", "trainer.train"]
