import numpy as np, os, sys
from collections import Counter
import pandas as pd
import random
import shutil

"""
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
"""

# Converting .psv file to np array
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data

def train_test_valid_files(path):
    files = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)
    
    random.shuffle(files)
    n_files = len(files)
    n_train = n_files * 6 // 10
    n_test = n_files * 2 // 10

    train_files = files[:n_train]
    test_files = files[n_train:n_train + n_test]
    valid_files = files[n_train + n_test:]
    return train_files, test_files, valid_files

path ="/data/Dataset"

train_files, test_files, valid_files = train_test_valid_files(path)

def add_file(files, directory_path):
    for f in files:
        source_file_path = os.path.join(path, f)
        destination_directory = new_directory_path
        shutil.copy(source_file_path, destination_directory)


##create a new training data direcory
new_directory_name = "Train_data"
data_folder = "data" 
new_directory_path = os.path.join(data_folder, new_directory_name)
directory_path = new_directory_path
shutil.rmtree(directory_path)
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
add_file(train_files, new_directory_path)


##create a new testing data direcory
new_directory_name = "Test_data"
data_folder = "data"  
new_directory_path = os.path.join(data_folder, new_directory_name)
directory_path = new_directory_path
shutil.rmtree(directory_path)
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
add_file(test_files, new_directory_path)


##create a new validation data direcory
new_directory_name = "Valid_data"
data_folder = "data" 
new_directory_path = os.path.join(data_folder, new_directory_name)
directory_path = new_directory_path
shutil.rmtree(directory_path)
if not os.path.exists(new_directory_path):
    os.makedirs(new_directory_path)
add_file(valid_files, new_directory_path)


### preprocessing 


#train
def train_preprocess_ffill_zero_imput(files):   #provide the list of train, test, valid files 
    num_columns = 38
    processed_data = np.empty((0, num_columns))
    
    for f in files:
        # Load data.
        input_file = os.path.join(path, f)
        data = load_challenge_data(input_file)
        data = pd.DataFrame(data)
        
        data = data.ffill()
        data = data.drop([7, 20, 27, 32], axis=1)
        data = data.fillna(0).values
        
        id = int(f[1:7])
        new_column = np.full((len(data), 1), id)
        data = np.hstack((new_column, data))
        
        processed_data = np.vstack((processed_data, data)) 
    processed_data = pd.DataFrame(processed_data)
    return processed_data
df1=train_preprocess_ffill_zero_imput(train_files)

def train_preprocesss_std(df):
    #num_columns = 38
    #data_list = np.empty((0, num_columns))

        
    columns_to_normalize = df.columns[1:-1]
    df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    mean_values = df[columns_to_normalize].mean()
    std_values = df[columns_to_normalize].std()
    mean_values.to_csv('mean_values.csv', header=False)
    std_values.to_csv('std_values.csv', header=False)
        
    #data_list = np.vstack((data_list, df)) 
    #data_list = pd.DataFrame(data_list)
    return df

train_preprocesss_std(df1)

def train_preprocesss_norm(df):
    columns_to_normalize = df.columns[1:-1]

    def min_max_scaling(column):
        min_val = column.min()
        max_val = column.max()
        scaled_column = (column - min_val) / (max_val - min_val)
        return scaled_column

    for col in columns_to_normalize:
        df[col] = min_max_scaling(df[col])
        #print(df[col])
    return df
train_preprocesss_norm(df1)

#test
def test_preprocess_ffill_zero_imput(files):   #provide the list of train, test, valid files 
    num_columns = 38
    processed_data = np.empty((0, num_columns))
    
    for f in files:
        # Load data.
        input_file = os.path.join(path, f)
        data = load_challenge_data(input_file)
        data = pd.DataFrame(data)
        
        data = data.ffill()
        data = data.drop([7, 20, 27, 32], axis=1)
        data = data.fillna(0).values
        
        id = int(f[1:7])
        new_column = np.full((len(data), 1), id)
        data = np.hstack((new_column, data))
        
        processed_data = np.vstack((processed_data, data)) 
    processed_data = pd.DataFrame(processed_data)
    return processed_data
df2=test_preprocess_ffill_zero_imput(test_files)

def test_preprocesss_std(df):

    mean_val = pd.read_csv("mean_values.csv", header=None, names=['mean'])
    std_val = pd.read_csv("std_values.csv", header=None, names=['std'])

    columns_to_normalize = df.columns[1:-1]
    df[columns_to_normalize] = (df[columns_to_normalize] - mean_val['mean'].values) / std_val['std'].values
    return df

test_preprocesss_std(df2)

def test_preprocesss_norm(df):
    columns_to_normalize = df.columns[1:-1]

    def min_max_scaling(column):
        min_val = column.min()
        max_val = column.max()
        scaled_column = (column - min_val) / (max_val - min_val)
        return scaled_column

    for col in columns_to_normalize:
        df[col] = min_max_scaling(df[col])
    return df
test_preprocesss_norm(df2)
