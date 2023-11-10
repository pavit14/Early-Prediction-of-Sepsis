import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np
from collections import Counter
import random
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf


@tf.function
def load_train_test_valid_files(**kwargs):
    """Loads the train, test, and validation data from the given path.

    Args:
        kwargs: A dictionary of arguments.

    Returns:
        A tuple of three tensors, containing the train, test, and validation data.
    """

    path = os.path.join(os.path.dirname(__file__), '../data/untitledFolder')
    files = tf.io.gfile.glob(os.path.join(path, '*.psv'))

    tf.random.shuffle(files)

    n_files = tf.shape(files)[0]
    n_train = n_files * 6 // 10
    n_test = n_files * 2 // 10

    train_files = files[:n_train]
    test_files = files[n_train:n_train + n_test]
    valid_files = files[n_train + n_test:]

    # Read the data from the files.
    train_data = tf.io.read_csv(train_files, sep='|')
    valid_data = tf.io.read_csv(valid_files, sep='|')
    test_data = tf.io.read_csv(test_files, sep='|')

    # Add the `id` column to the data.
    train_data['id'] = tf.range(tf.shape(train_data)[0])
    valid_data['id'] = tf.range(tf.shape(valid_data)[0])
    test_data['id'] = tf.range(tf.shape(test_data)[0])

    # Remove the last column from the data.
    train_data = train_data.drop(train_data.columns[-1], axis=1)
    valid_data = valid_data.drop(valid_data.columns[-1], axis=1)
    test_data = test_data.drop(test_data.columns[-1], axis=1)

    # Serialize the data to TensorFlow tensors.
    train_serialized_data = tf.io.serialize_tensor(train_data)
    valid_serialized_data = tf.io.serialize_tensor(valid_data)
    test_serialized_data = tf.io.serialize_tensor(test_data)

    # Push the serialized data to xcom.
    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

    return train_serialized_data, valid_serialized_data, test_serialized_data

@tf.function
def feature_engineering(**kwargs):
    """Performs feature engineering on the given data.

    Args:
        kwargs: A dictionary of arguments.

    Returns:
        A tuple of three tensors, containing the train, test, and validation data with
        feature engineering applied.
    """

    ti = kwargs['ti']

    # Pull the serialized data from xcom.
    train_serialized_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='train_data')
    valid_serialized_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='valid_data')
    test_serialized_data = ti.xcom_pull(task_ids='load_train_test_valid_files', key='test_data')

    # Deserialize the data to TensorFlow tensors.
    train_df = pd.DataFrame(pickle.loads(train_serialized_data))
    valid_df = pd.DataFrame(pickle.loads(valid_serialized_data))
    test_df = pd.DataFrame(pickle.loads(test_serialized_data))

    # Perform feature engineering on the data.
    train_df = train_df.groupby('id').filter(lambda x: len(x) >= 15)
    train_df = train_df.groupby('id').ffill()
    train_df = train_df.drop(train_df.columns[[8, 21, 28, 33]], axis=1)

    valid_df = valid_df.groupby('id').ffill()
    valid_df = valid_df.drop(valid_df.columns[[8, 21, 28, 33]], axis=1)

    test_df = test_df.groupby('id').ffill()
    test_df = test_df.drop(test_df.columns[[8, 21, 28, 33]], axis=1)

    # Serialize the data to TensorFlow tensors.
    train_serialized_data = tf.io.serialize_tensor(train_df)
    valid_serialized_data = tf.io.serialize_tensor(valid_df)
    test_serialized_data = tf.io.serialize_tensor(test_df)

    # Push the serialized data to xcom.
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

    return train_serialized_data, valid_serialized_data, test_serialized_data



@tf.function
def preprocess_zero_imput_norm(**kwargs):
    """Imputes missing values with zeros and normalizes the data.

    Args:
        kwargs: A dictionary of arguments.

    Returns:
        A tuple of three tensors, containing the train, test, and validation data with
        missing values imputed with zeros and normalized.
    """

    ti = kwargs['ti']

    # Pull the serialized data from xcom.
    train_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='train_data')
    valid_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='valid_data')
    test_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='test_data')

    # Deserialize the data to TensorFlow tensors.
    train_df = pd.DataFrame(pickle.loads(train_serialized_data))
    valid_df = pd.DataFrame(pickle.loads(valid_serialized_data))
    test_df = pd.DataFrame(pickle.loads(test_serialized_data))

    # Impute missing values with zeros.
    train_df = train_df.fillna(0)
    valid_df = valid_df.fillna(0)
    test_df = test_df.fillna(0)

    # Normalize the data.
    scaler = MinMaxScaler()
    train_df.iloc[:, 1:-1] = scaler.fit_transform(train_df.iloc[:, 1:-1])
    valid_df.iloc[:, 1:-1] = scaler.transform(valid_df.iloc[:, 1:-1])
    test_df.iloc[:, 1:-1] = scaler.transform(test_df.iloc[:, 1:-1])

    # Save the preprocessed data to files.
    train_df.to_csv(os.path.join(os.path.dirname(__file__), '../data/', 'my_train_data.csv'), index=False)
    valid_df.to_csv(os.path.join(os.path.dirname(__file__), '../data/', 'my_valid_data.csv'), index=False)
    test_df.to_csv(os.path.join(os.path.dirname(__file__), '../data/', 'my_test_data.csv'), index=False)

    # Serialize the preprocessed data to TensorFlow tensors.
    train_serialized_data = pickle.dumps(train_df)
    valid_serialized_data = pickle.dumps(valid_df)
    test_serialized_data = pickle.dumps(test_df)

    # Push the serialized data to xcom.
    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

    return train_serialized_data, valid_serialized_data, test_serialized_data


@tf.function
def preprocess_mean_input_norm(**kwargs):
    """Imputes missing values with the mean and normalizes the data.

    Args:
        kwargs: A dictionary of arguments.

    Returns:
        A tuple of three tensors, containing the train, test, and validation data with
        missing values imputed with the mean and normalized.
    """

    ti = kwargs['ti']

    # Pull the serialized data from xcom.
    train_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='train_data')
    valid_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='valid_data')
    test_serialized_data = ti.xcom_pull(task_ids='feature_engineering', key='test_data')

    # Deserialize the data to TensorFlow tensors.
    train_df = pd.DataFrame(pickle.loads(train_serialized_data))
    valid_df = pd.DataFrame(pickle.loads(valid_serialized_data))
    test_df = pd.DataFrame(pickle.loads(test_serialized_data))

    # Impute missing values with the mean.
    mean_values = train_df.mean(axis=0)
    train_df = train_df.fillna(mean_values)
    valid_df = valid_df.fillna(mean_values)
    test_df = test_df.fillna(mean_values)

    # Normalize the data.
    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    valid_df = scaler.transform(valid_df)
    test_df = scaler.transform(test_df)

    # Serialize the preprocessed data to TensorFlow tensors.
    train_serialized_data = tf.io.serialize_tensor(train_df)
    valid_serialized_data = tf.io.serialize_tensor(valid_df)
    test_serialized_data = tf.io.serialize_tensor(test_df)

    # Push the serialized data to xcom.
    ti = kwargs['ti']
    ti.xcom_push(key='train_data', value=train_serialized_data)
    ti.xcom_push(key='valid_data', value=valid_serialized_data)
    ti.xcom_push(key='test_data', value=test_serialized_data)

    return train_serialized_data, valid_serialized_data, test_serialized_data