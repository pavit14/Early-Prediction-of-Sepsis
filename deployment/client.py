import requests
import numpy as np
import base64

# Create a NumPy array
with open('/Users/anurag/Desktop/Snehith2/X_test.npy', "rb") as f:
    arr = np.load(f)
# arr = np.array([1, 2, 3, 4], dtype=np.float32)

# Convert the array to base64
arr_b64 = base64.b64encode(arr).decode()

# Send the array to the FastAPI service
response = requests.post("http://127.0.0.1:8000/api/v1/sepsis/predict", json={"array": arr_b64})

# print(response.json())
# Decode the response back into a NumPy array
result_arr = np.frombuffer(base64.b64decode(response.json()['predictions']), dtype=np.float32)

print(result_arr)
