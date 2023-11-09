import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split


def load_train_test_valid_files(**kwargs):
    files = []
    path = os.path.join(os.path.dirname(__file__), '../data/Dataset')
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


    train_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in train_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=int(file_id[1:])
            data['id'] = file_id
            train_df = pd.concat([train_df, data], ignore_index=True)

    tr_last_column = train_df.pop(train_df.columns[-1])  # Remove the last column
    train_df.insert(0, tr_last_column.name, tr_last_column)

    test_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in test_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=file_id[1:]
            data['id'] = file_id
            test_df = pd.concat([test_df, data], ignore_index=True)

    te_last_column = test_df.pop(test_df.columns[-1])  # Remove the last column
    test_df.insert(0, te_last_column.name, te_last_column)

    valid_df = pd.DataFrame()
    for filename in os.listdir(path):
        if filename.endswith('.psv') and filename in valid_files:
            file_path = os.path.join(path, filename)
            data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
            file_id = os.path.splitext(filename)[0]
            file_id=file_id[1:]
            data['id'] = file_id
            valid_df = pd.concat([valid_df, data], ignore_index=True)
    
    va_last_column = valid_df.pop(valid_df.columns[-1])  # Remove the last column
    valid_df.insert(0, va_last_column.name, va_last_column)

    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

def feature_engineering(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='train_data')
    valid_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='valid_data')
    test_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='test_data')

    train_df = pickle.loads(train_data)
    train_df = train_df.groupby('id').filter(lambda x: len(x) >=15 )
    train_df= train_df.groupby('id').ffill()
    train_df = train_df.drop(train_df.columns[[8, 21, 28, 33]], axis=1)
    
    valid_df = pickle.loads(valid_data)
    valid_df= valid_df.groupby('id').ffill()
    valid_df = valid_df.drop(valid_df.columns[[8, 21, 28, 33]], axis=1)

    test_df = pickle.loads(test_data)
    test_df= test_df.groupby('id').ffill()
    test_df = test_df.drop(test_df.columns[[8, 21, 28, 33]], axis=1)

    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

def preprocess_zero_imput_norm(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='feature_engineering', key='train_data')
    valid_data = ti.xcom_pull(task_ids='feature_engineering', key='valid_data')
    test_data = ti.xcom_pull(task_ids='feature_engineering', key='test_data')

    train_df = pickle.loads(train_data)
    test_df = pickle.loads(test_data)
    valid_df = pickle.loads(valid_data)

    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)
    valid_df = valid_df.fillna(0)

    scaler = MinMaxScaler()
    train_df.iloc[:,1:-1] = scaler.fit_transform(train_df.iloc[:,1:-1])
    test_df.iloc[:,1:-1] = scaler.transform(test_df.iloc[:,1:-1])
    valid_df.iloc[:,1:-1] = scaler.transform(valid_df.iloc[:, 1:-1])


    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

def preprocess_mean_input_norm(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='feature_engineering', key='train_data')
    valid_data = ti.xcom_pull(task_ids='feature_engineering', key='valid_data')
    test_data = ti.xcom_pull(task_ids='feature_engineering', key='test_data')

    train_df = pickle.loads(train_data)
    test_df = pickle.loads(test_data)
    valid_df = pickle.loads(valid_data)

    mean_values = train_df.mean()
    train_df.fillna(mean_values, inplace=True)
    test_df.fillna(mean_values, inplace=True)
    valid_df.fillna(mean_values, inplace=True)
    
    scaler = MinMaxScaler()
    train_df.iloc[:,1:-1] = scaler.fit_transform(train_df.iloc[:,1:-1])
    test_df.iloc[:,1:-1] = scaler.transform(test_df.iloc[:,1:-1])
    valid_df.iloc[:,1:-1] = scaler.transform(valid_df.iloc[:, 1:-1])

    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)
