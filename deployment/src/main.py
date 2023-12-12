# uvicorn src.main:app --reload 

from fastapi import FastAPI
import pickle

app = FastAPI()

MODEL_PATH = "/Users/anurag/Downloads/mlruns/0/28b3e2b4bda84ccf8c1ddbcf96acc41c/artifacts/model1/model.pkl"

def load_model():
    with open(MODEL_PATH, "rb") as f:
        model.load_weights(pickle.load(f))
    return model

model = load_model()

@app.get("/hello")
async def root():
    return {"message": "Hello World"}


@app.get("/api/v1/sepsis/predict")
async def predict():
    return {"message": "Hello World"}
