from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import random
import pandas as pd
import numpy as np, os, sys
import sys
#sys.path.append("D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis")
sys.path.append("..")
#from src.Load_Data import preprocess_zero_imput, preprocess_ffill, train_test_valid_files, load_challenge_data


def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data

def train_test_valid_files(shuffle_files = True):
    files = []
    path = "D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data\Downloaded_data"
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
    print(train_files, test_files, valid_files)
    return train_files, test_files, valid_files

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





def zero_imputation(**kwargs):
    ti = kwargs['ti']
    output_lists = ti.xcom_pull(task_ids='train_test_valid_split')
    train_files, test_files, valid_files = output_lists

    zero_ffill_train = preprocess_zero_imput(preprocess_ffill(train_files))
    zero_ffill_test = preprocess_zero_imput(preprocess_ffill(test_files))
    zero_ffill_valid = preprocess_zero_imput(preprocess_ffill(valid_files))

    file_path_train = os.path.join('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data', '_train_data.csv')
    file_path_test = os.path.join('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data', '_test_data.csv')
    file_path_valid = os.path.join('D:\IE7374_MLOps\Final_project\Early-Prediction-of-Sepsis\data', '_valid_data.csv')
    zero_ffill_train.to_csv(file_path_train, index=False)
    zero_ffill_test.to_csv(file_path_test, index=False)
    zero_ffill_valid.to_csv(file_path_valid, index=False)

    return zero_ffill_train, zero_ffill_test, zero_ffill_valid



with DAG("Data_pipeline_dag", start_date=datetime(2023, 1, 1), 
    schedule= timedelta(seconds=30), catchup=False) as dag:
    
    train_test_valid_split = PythonOperator(
        task_id="train_test_valid_split",
        python_callable=train_test_valid_files,
        #provide_context=True,
        dag=dag,
    )

    zero_imput_files = PythonOperator(
        task_id="zero_imput_files", 
        python_callable=zero_imputation,
        #provide_context=True,
        dag=dag,
    )

    train_test_valid_split >> zero_imput_files

    