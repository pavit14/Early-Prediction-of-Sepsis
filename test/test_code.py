import numpy as np, os, sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
from src.Load_Data import load_challenge_data
import pandas as pd

"""
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data
"""

files = []
for f in os.listdir('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data'):
    if os.path.isfile(os.path.join('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data', f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
        files.append(f)

print(files)

input_file = os.path.join("D:\IE7374_MLOps\Project_Datasets\Practice\Dataset", files[1])
data = load_challenge_data(input_file)
print(data)



