import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from kneed import KneeLocator
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split

'''
def load_data():
    path =os.path.join(os.path.dirname(__file__), '../data/Dataset')
    df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv'):
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            data['id'] = file_id
            df = pd.concat([df, data], ignore_index=True)

    serialized_data = pickle.dumps(df)
    return serialized_data 

def train_test_val_split(data):
    df = pickle.loads(data)
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    valid_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)
    return train_serialized_data,valid_serialized_data,test_serialized_data
'''

def preprocess_train_data(**kwargs):
    #ti = kwargs['ti']
    #train_data, valid_data, test_data = ti.xcom_pull(task_ids='train_test_valid_files')
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='train_test_valid_files', key='train_data')
    valid_data = ti.xcom_pull(task_ids='train_test_valid_files', key='valid_data')
    test_data = ti.xcom_pull(task_ids='train_test_valid_files', key='test_data')
    
    #output_param = kwargs['params']['output_param']
    #train_data, valid_data, test_data = kwargs['params'][output_param]
    df = pickle.loads(train_data)
    df = df.ffill()
    df = df.drop(df.columns[[7, 20, 27, 32]], axis=1)
    columns_to_normalize = df.columns[:-2]
    df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].mean()) / df[columns_to_normalize].std()
    mean_values = df[columns_to_normalize].mean()
    mean_values_df = mean_values.reset_index()
    mean_values_df.columns = ['Column_Name', 'Mean']
    std_values = df[columns_to_normalize].std()
    std_values_df = std_values.reset_index()
    std_values_df.columns = ['Column_Name', 'Standard_Deviation']
    #mean_values.to_csv('mean_values.csv', header=False)
    #std_values.to_csv('std_values.csv', header=False)
    mean_values_data = pickle.dumps(mean_values_df)
    std_values_data = pickle.dumps(std_values_df)


    ti.xcom_push(key='mean_values_data', value=mean_values_data)
    ti.xcom_push(key='std_values_data', value=std_values_data)
    ti.xcom_push(key='valid_data', value=valid_data)
    ti.xcom_push(key='test_data', value=test_data)

    #return mean_values_data,std_values_data,valid_data,test_data

def preprocess_test_val_data(**kwargs):
    #ti = kwargs['ti']
    #mean_values_data,std_values_data,valid_data,test_data = ti.xcom_pull(task_ids='train_test_valid_files')
    ti = kwargs['ti']
    mean_values_data = ti.xcom_pull(task_ids='preprocess_train_data', key='mean_values_data')
    std_values_data = ti.xcom_pull(task_ids='preprocess_train_data', key='std_values_data')
    valid_data = ti.xcom_pull(task_ids='preprocess_train_data', key='valid_data')
    test_data = ti.xcom_pull(task_ids='preprocess_train_data', key='test_data')

    df_valid=pickle.loads(valid_data)
    df_test=pickle.loads(test_data)
    df_mean=pickle.loads(mean_values_data)
    df_std=pickle.loads(std_values_data)

    df_valid = df_valid.drop(df_valid.columns[[7, 20, 27, 32]], axis=1)
    df_test = df_test.drop(df_test.columns[[7, 20, 27, 32]], axis=1)

    val_columns_to_normalize = df_valid.columns[:-2]
    test_columns_to_normalize = df_test.columns[:-2]

    df_valid[val_columns_to_normalize] = (df_valid[val_columns_to_normalize] - df_mean['Mean'].values) / df_std['Standard_Deviation'].values
    df_test[test_columns_to_normalize] = (df_test[test_columns_to_normalize] - df_mean['Mean'].values) / df_std['Standard_Deviation'].values
    valid_data = pickle.dumps(df_valid)
    test_data = pickle.dumps(df_test)
    return valid_data,test_data


def train_test_valid_files(**kwargs):
    shuffle_files = True
    files = []
    path = os.path.join(os.path.dirname(__file__), '../data/Dataset')
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


    train_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in train_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            data['id'] = file_id
            train_df = pd.concat([train_df, data], ignore_index=True)


    test_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in test_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            data['id'] = file_id
            test_df = pd.concat([test_df, data], ignore_index=True)


    valid_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in valid_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            data['id'] = file_id
            valid_df = pd.concat([valid_df, data], ignore_index=True)
    

    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    print(train_serialized_data)
    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)
    #return {'train_data': train_serialized_data, 'valid_data': valid_serialized_data, 'test_data': test_serialized_data}
    #return train_serialized_data,valid_serialized_data,test_serialized_data

'''
a,b,c=train_test_valid_files
d,e,b,c=preprocess_train_data(a,b,c)
y,z=preprocess_test_val_data(d,e,b,c)
'''