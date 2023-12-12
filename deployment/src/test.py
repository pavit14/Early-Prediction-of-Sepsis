import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

MODEL_PATH      = "/Users/anurag/Desktop/Snehith2/model_weights.h5"

MAX_LEN         = 50
DIM             = 39
NUM_NEURONS     = 50
EPOCHS          = 60
BATCH_SIZE      = 64
DROP_RATE       = 0.2
OPTIMIZER       = 'RMSprop'

model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(NUM_NEURONS, input_shape=(MAX_LEN, DIM), return_sequences=True))
model.add(tf.keras.layers.Dropout(DROP_RATE))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=OPTIMIZER, metrics=['accuracy'])
model.load_weights(MODEL_PATH)

def test(X_test):
    tf.keras.backend.set_image_data_format("channels_last")
    
    # model = tf.keras.Sequential()
    X_test = np.load(f)
    # print(X_test.shape)
    # print(X_test[0])
    # print(X_test[0].shape)
    predictions = model.predict(X_test)
    # print(predictions)
    return predictions
    # print(model.summary())

with open('/Users/anurag/Desktop/Snehith2/X_test.npy', "rb") as f:
    test(f)



# import pickle

# # Replace 'your_model.pkl' with the actual path to your pickle file
# pickle_file_path = '/Users/anurag/Downloads/mlruns/0/28b3e2b4bda84ccf8c1ddbcf96acc41c/artifacts/model1/model.pkl'

# # Load the model from the pickle file
# with open(pickle_file_path, 'rb') as file:
#     loaded_model = pickle.load(file)

# # Now 'loaded_model' contains your model, and you can use it for predictions or other tasks