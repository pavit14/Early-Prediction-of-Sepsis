import numpy as np, os, sys
from collections import Counter
import pandas as pd
import random
import shutil
from sklearn.preprocessing import MinMaxScaler

"""
from sklearn.metrics import confusion_matrix
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

def train_test_valid_files(path, shuffle_files = False):
    files = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
            files.append(f)
    
    if shuffle_files == True:
        random.shuffle(files)

    n_files = len(files)
    n_train = n_files * 6 // 10
    n_test = n_files * 2 // 10

    train_files = files[:n_train]
    test_files = files[n_train:n_train + n_test]
    valid_files = files[n_train + n_test:]
    return train_files, test_files, valid_files

##path ="D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data"

#train_files, test_files, valid_files = train_test_valid_files(path, True)
"""
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
"""

### preprocessing 
def preprocess_ffill(files):   #provide the list of train, test, valid files 
    num_columns = 38
    ffill_data = np.empty((0, num_columns))
    
    for f in files:
        # Load data.
        input_file = os.path.join(path, f)
        path = "D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data"
        data = load_challenge_data(input_file)
        data = pd.DataFrame(data)
        
        data = data.ffill()
        data = data.drop([7, 20, 27, 32], axis=1)

        id = int(f[1:7])
        new_column = np.full((len(data), 1), id)
        data = np.hstack((new_column, data))
        
        ffill_data = np.vstack((ffill_data, data)) 
    ffill_data = pd.DataFrame(ffill_data)
    print(ffill_data)
    return ffill_data

def preprocess_zero_imput(df):
    df = df.fillna(0)
    return df

def preprocess_mean_imput(df, train_ffill_df):
    column_means = train_ffill_df.mean()
    df_imputed = df.fillna(column_means)
    return df

"""
train_df = preprocess_ffill(train_files)
test_df = preprocess_ffill(test_files)
df = preprocess_mean_imput(test_df, train_df)
file_path = os.path.join('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data', 'my_train_data.csv')
df.to_csv(file_path, index=False)
"""



"""
# using forward filling for filling the null values. input and output in terms of df
def f_fill(data):
    data = data.ffill()
    return data

# filling null values with 0 
def fill_zero(data):
    data = data.fillna(0)
    return data

# parsing patient id of patient
def patient_id(file):
    id = int(file[1:7])
    return id

# dropping the column in data
def drop_column(data, col_num):
    data = data.drop(col_num, axis=1)
    return data


def create_sequence_data(df, columns):
    
    id_list = df[0].unique()
    
    x_list = []
    y_list = []
    ids = []

    for id in id_list:
        x_list.append(df[df[0] == id][columns].values.tolist())
        y_list.append(df[df[0] == id][37].max())
        ids.append(id)

    return x_list, y_list, ids

def pad_trunc(data, max_len):
    new_data = []
    zero_vec = []
    for _ in range(len(data[0][0])):
        zero_vec.append(0.0)
    
    for sample in data:
        if len(sample) > max_len: # truncate 
            temp = sample[:max_len]
        elif len(sample) < max_len:
            temp = sample
            number_additonal_elements = max_len - len(sample)
            for _ in range(number_additonal_elements):
                temp.append(zero_vec)
        else:
            temp = sample
    
        new_data.append(temp)
    
    return new_data
    
# return the patient files in folder 
files = []
for f in os.listdir('D:\IE7374_MLOps\Project_Datasets\Practice\Files\Training_set_A'):
    if os.path.isfile(os.path.join('D:\IE7374_MLOps\Project_Datasets\Practice\Files\Training_set_A', f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
        files.append(f)


num_columns = 38
data_list = np.empty((0, num_columns))
len_data = []

for f in files:
    # Load data.
    input_file = os.path.join("D:\IE7374_MLOps\Project_Datasets\Practice\Dataset", f)
    data = load_challenge_data(input_file)
    data = pd.DataFrame(data)
    data = f_fill(data)
    data = data.drop([7, 20, 27, 32], axis=1)
    data = fill_zero(data).values
    
    id = patient_id(f)
    
    new_column = np.full((len(data), 1), id)
    data = np.hstack((new_column, data))
    
    #len_data.append(len(data))
    data_list = np.vstack((data_list, data)) 
data_list

train_list = data_list[data_list[:,0]<= 20643]

test_condition = np.logical_and(data_list[:, 0] > 20643, data_list[:, 0] <= 30643)
test_list = data_list[test_condition]

valid_list = data_list[data_list[:,0]>30643]

train_df = pd.DataFrame(train_list)
test_df = pd.DataFrame(test_list)
valid_df = pd.DataFrame(valid_list)

feature_columns = train_df.iloc[:, :-1].columns

X_train, y_train, id_ls_train = create_sequence_data(train_df, feature_columns)
X_test, y_test, id_ls_test = create_sequence_data(test_df, feature_columns)
X_valid, y_valid, id_ls_test = create_sequence_data(valid_df, feature_columns)


maxLen = 0
for i in X_train:
    maxLen = max(maxLen, len(i))

max_len = maxLen
batch_size = 32
embedding_dim = 37

X_train = pad_trunc(X_train, max_len)
X_test = pad_trunc(X_test, max_len)
X_valid = pad_trunc(X_valid, max_len)

# Reshape x_train and x_test
X_train = np.reshape(X_train, (len(X_train), max_len, embedding_dim))
X_test = np.reshape(X_test, (len(X_test), max_len, embedding_dim))
X_valid = np.reshape(X_valid, (len(X_valid), max_len, embedding_dim))

y_train = np.array(y_train)
y_test = np.array(y_test)
y_valid = np.array(y_valid)

"""




