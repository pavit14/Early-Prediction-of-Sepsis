import numpy as np, os, sys
from collections import Counter
import pandas as pd
import random
import shutil

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

def add_file(files, directory_path):
    for f in files:
        source_file_path = os.path.join(path, f)
        destination_directory = new_directory_path
        shutil.copy(source_file_path, destination_directory)

### preprocessing 

def preprocess_ffill(files):   #provide the list of train, test, valid files 
    num_columns = 38
    ffill_data = np.empty((0, num_columns))
    
    for f in files:
        # Load data.
        input_file = os.path.join(path, f)
        data = load_challenge_data(input_file)
        data = pd.DataFrame(data)
        
        data = data.ffill()
        data = data.drop([7, 20, 27, 32], axis=1)

        id = int(f[1:7])
        new_column = np.full((len(data), 1), id)
        data = np.hstack((new_column, data))
        
        ffill_data = np.vstack((ffill_data, data)) 
    ffill_data = pd.DataFrame(ffill_data)
    return ffill_data

def preprocess_zero_imput(df):
    df = df.fillna(0)
    return df

def preprocess_mean_imput(df, train_ffill_df):
    column_means = train_ffill_df.mean()
    df_imputed = df.fillna(column_means)
    return df

def train_preprocesss_std(df):
    columns_to_normalize = df.columns[1:-1]
    df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    mean_values = df[columns_to_normalize].mean()
    std_values = df[columns_to_normalize].std()
    mean_values.to_csv('mean_values.csv', header=False)
    std_values.to_csv('std_values.csv', header=False)
    return df

def preprocesss_norm(df):
    columns_to_normalize = df.columns[1:-1]

    def min_max_scaling(column):
        min_val = column.min()
        max_val = column.max()
        scaled_column = (column - min_val) / (max_val - min_val)
        return scaled_column

    for col in columns_to_normalize:
        df[col] = min_max_scaling(df[col])
    return df

def test_preprocesss_std(df):

    mean_val = pd.read_csv("mean_values.csv", header=None, names=['mean'])
    std_val = pd.read_csv("std_values.csv", header=None, names=['std'])

    columns_to_normalize = df.columns[1:-1]
    df[columns_to_normalize] = (df[columns_to_normalize] - mean_val['mean'].values) / std_val['std'].values
    return df


path ="data/Dataset"
train_files, test_files, valid_files = train_test_valid_files(path)

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

train_df = preprocess_ffill(train_files)
train_std=train_preprocesss_std(train_df)
train_norm=preprocesss_norm(train_df)
test_df = preprocess_ffill(test_files)
test_std=test_preprocesss_std(test_df)
test_norm=preprocesss_norm(test_df)
df = preprocess_mean_imput(test_df, train_df)
