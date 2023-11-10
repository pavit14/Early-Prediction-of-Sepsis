# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow import configuration as conf
#from src.pylab import preprocess_train_data,preprocess_test_val_data,train_test_valid_files

from airflow import configuration as conf

import pandas as pd
#from sklearn.preprocessing import MinMaxScaler
#from kneed import KneeLocator
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
#from sklearn.model_selection import train_test_split



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
    path = os.path.join(os.path.dirname(__file__), '../../data/Downloaded_data')
    #path =os.path.join(os.path.dirname(__file__), '../data/Dataset')
    #path = "D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data"
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






# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Create a DAG instance named 'your_python_dag' with the defined default arguments
dag = DAG(
    'early_sepsis_prediction',
    default_args=default_args,
    description='Data Pipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function

# Define your Airflow tasks

train_test_valid_files_task = PythonOperator(
    task_id='train_test_valid_files',
    python_callable=train_test_valid_files,
    provide_context=True,
    #op_kwargs={'output_param': 'output_result'},
    dag=dag
)


preprocess_train_data_task = PythonOperator(
    task_id='preprocess_train_data',
    python_callable=preprocess_train_data,
    provide_context=True,
    #op_kwargs={'output_param': 'output_result'},
    dag=dag
)

preprocess_test_val_data_task = PythonOperator(
    task_id='preprocess_test_val_data',
    python_callable=preprocess_test_val_data,
    provide_context=True,
    dag=dag
)

# Set task dependencies
train_test_valid_files_task >> preprocess_train_data_task >> preprocess_test_val_data_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()

