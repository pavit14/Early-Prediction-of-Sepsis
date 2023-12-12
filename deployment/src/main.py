# uvicorn src.main:app --reload 

from fastapi import FastAPI
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from .constants import *
from pydantic import BaseModel
import base64

app = FastAPI()

# MODEL_PATH = "/Users/anurag/Downloads/mlruns/0/28b3e2b4bda84ccf8c1ddbcf96acc41c/artifacts/model1/model.pkl"
model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(NUM_NEURONS, input_shape=(MAX_LEN, DIM), return_sequences=True))
model.add(tf.keras.layers.Dropout(DROP_RATE))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.load_weights(MODEL_PATH)



class PayloadData(BaseModel):
    array: str

@app.get("/hello")
async def root():
    return {"message": "Hello World"}


@app.post("/api/v1/sepsis/predict")
async def predict(data: PayloadData):
    arr = np.frombuffer(base64.b64decode(data.array), dtype=np.float32)
    arr = arr.reshape(-1, MAX_LEN, DIM)
    predictions = model.predict(arr)
    result = base64.b64encode(predictions).decode()
    return {"predictions": result}
