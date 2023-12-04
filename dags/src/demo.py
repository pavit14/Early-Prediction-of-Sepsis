import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def load_train_files(**kwargs):
        i=0
        train_files = []
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
        for f in os.listdir(path):
            if i>200000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                train_files.append(f)
                i=i+1

        train_df = pd.DataFrame()
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setA')
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
        train_serialized_data = pickle.dumps(train_df)
        ti = kwargs['ti']
        ti.xcom_push(key='train_data', value=train_serialized_data)


def load_test_files(**kwargs):
        
        ti = kwargs['ti']
        train_data = ti.xcom_pull(task_ids='load_train_files', key='train_data')
        train_df = pickle.loads(train_data)
        i=0
        test_files = []
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB')
        for f in os.listdir(path):
            if i>10000:
                break
            if os.path.isfile(os.path.join(path, f)) and not f.lower().startswith('.') and f.lower().endswith('psv'):
                test_files.append(f)
                i=i+1

        test_df = pd.DataFrame()
        path = os.path.join(os.path.dirname(__file__), '../data/Dataset/physionet.org/files/challenge-2019/1.0.0/training/training_setB')
        for filename in os.listdir(path):
            if filename.endswith('.psv') and filename in test_files:
                file_path = os.path.join(path, filename)
                data = pd.read_csv(file_path, sep='|')  # Assuming the files are pipe-separated
                file_id = os.path.splitext(filename)[0]
                file_id=int(file_id[1:])
                data['id'] = file_id
                test_df = pd.concat([test_df, data], ignore_index=True)

        ts_last_column = test_df.pop(test_df.columns[-1])  # Remove the last column
        test_df.insert(0, ts_last_column.name, ts_last_column)
        test_serialized_data = pickle.dumps(test_df)
        ti = kwargs['ti']
        ti.xcom_push(key='test_data', value=test_serialized_data)
        train_serialized_data = pickle.dumps(train_df)
        ti.xcom_push(key='train_data', value=train_serialized_data) 

def feature_engineering(**kwargs):
    ti = kwargs['ti']
    train_data = ti.xcom_pull(task_ids='load_train_files', key='train_data')
    test_data = ti.xcom_pull(task_ids='load_test_files', key='test_data')

    train_df = pickle.loads(train_data)
    train_df = train_df.drop(train_df.columns[[37,38]], axis=1)
    column_index_to_encode = 36
    encoded_columns = pd.get_dummies(train_df.iloc[:, column_index_to_encode], prefix='column_36')
    train_df.drop(train_df.columns[column_index_to_encode], axis=1, inplace=True)
    train_df = pd.concat([train_df, encoded_columns], axis=1)
    train_df = train_df.drop('EtCO2', axis=1)

    test_df = pickle.loads(test_data)
    test_df = test_df.drop(test_df.columns[[37,38]], axis=1)
    encoded_columns = pd.get_dummies(test_df.iloc[:, column_index_to_encode], prefix='column_36')
    test_df.drop(test_df.columns[column_index_to_encode], axis=1, inplace=True)
    test_df = pd.concat([test_df, encoded_columns], axis=1)
    test_df = test_df.drop('EtCO2', axis=1)

    threshold = 1.0  # 90%
    threshold_count = len(train_df) * threshold
    columns_to_drop = train_df.columns[train_df.isnull().sum() >= threshold_count]
    train_df.drop(columns=columns_to_drop, inplace=True, axis=1)
    test_df.drop(columns=columns_to_drop, inplace=True, axis=1)

    if len(train_df) > 1:
        mask = train_df.groupby('id')['id'].transform('count') >= 15
        train_df = train_df[mask]
        train_df = train_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    

    if len(test_df) > 1:
        test_df = test_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)

    mean_values = train_df.mean()
    mean_values.to_csv('/Users/pavithra/Desktop/xyz/dags/mean_values.csv')
    test_df.fillna(mean_values, inplace=True)    
    train_df.fillna(mean_values, inplace=True)
    
    train_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/m_train_df.pkl')
    test_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/m_test_df.pkl')

    X = train_df.iloc[:, 1:-1]
    y=train_df.iloc[:, -1]

    train_set, valid_set, train_labels, valid_labels = train_test_split(X, y, test_size=0.33, random_state=42)

    rf_clf = RandomForestClassifier()
    rf_clf.fit(train_set, train_labels)

    predictions = rf_clf.predict(valid_set)
    feature_impt = pd.DataFrame(rf_clf.feature_importances_, index=train_set.columns)
    non_impt_features = feature_impt.loc[feature_impt[0] <= 0.005]
    non_impt_features_list = list(non_impt_features.index.values)
    #non_impt_column_indices = [train_df.columns.get_loc(col) for col in non_impt_features_list]

    train_df = train_df.drop(non_impt_features_list, axis=1)
    test_df = test_df.drop(non_impt_features_list, axis=1)


    train_serialized_data = pickle.dumps(train_df)
    test_serialized_data = pickle.dumps(test_df)

    train_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/train_df.pkl')
    test_df.to_pickle('/Users/pavithra/Desktop/xyz/dags/test_df.pkl')

    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)



