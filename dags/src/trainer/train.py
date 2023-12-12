from google.cloud import storage
from datetime import datetime
import pytz
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
import joblib
import gcsfs
import os
from dotenv import load_dotenv

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os
import numpy as np

# Load environment variables
load_dotenv()

# Initialize variables
fs = gcsfs.GCSFileSystem()
storage_client = storage.Client()
bucket_name = os.getenv("BUCKET_NAME")
MODEL_DIR = os.environ['AIP_STORAGE_URI']


def load_train_data(gcs_train_data_path):

    with fs.open(gcs_train_data_path) as f:
        train_df = pd.read_csv(f)

    return train_df

def load_test_data(gcs_test_data_path):

    with fs.open(gcs_test_data_path) as f:
        test_df = pd.read_csv(f)

    return test_df


def feature_engineering(train_df,test_df):

    train_df = train_df.drop(train_df.columns[[37,38]], axis=1)
    column_to_encode = 'Gender'
    encoded_columns = pd.get_dummies(train_df[column_to_encode], prefix=column_to_encode) 
    position = train_df.columns.get_loc(column_to_encode) 
    train_df.drop(column_to_encode, axis=1, inplace=True) 
    train_df = pd.concat([train_df.iloc[:, :position], encoded_columns, train_df.iloc[:, position:]], axis=1)


    test_df = test_df.drop(test_df.columns[[37,38]], axis=1)
    encoded_columns = pd.get_dummies(test_df[column_to_encode], prefix=column_to_encode) 
    position = test_df.columns.get_loc(column_to_encode) 
    test_df.drop(column_to_encode, axis=1, inplace=True) 
    test_df = pd.concat([test_df.iloc[:, :position], encoded_columns, test_df.iloc[:, position:]], axis=1)

    train_df = train_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)
    test_df = test_df.groupby('id').apply(lambda x: x.ffill()).reset_index(drop=True)

    return train_df,test_df

def mean_imputation(train_df,test_df):


    def delete_last_5_records(group):
        return group.iloc[:-5]

    def select_last_50_or_all(group):
        return group.tail(50) if len(group) >= 50 else group

    train_df = train_df.groupby(train_df['id'], group_keys=False).apply(delete_last_5_records)
    train_df = train_df.groupby(train_df['id'], group_keys=False).apply(select_last_50_or_all)
    test_df = test_df.groupby(test_df['id'], group_keys=False).apply(delete_last_5_records)
    test_df = test_df.groupby(test_df['id'], group_keys=False).apply(select_last_50_or_all)

    mean_values = train_df.mean()

    serialized_mean = pickle.dumps(mean_values)
    local_model_path = "mean.pkl"
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    gcs_model_path = f"{MODEL_DIR}/mean_{version}.pkl"
    # Save the model locally
    joblib.dump(serialized_mean, local_model_path)

    # Upload the model to GCS
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)

    ##

    train_df = train_df.fillna(mean_values)
    test_df = test_df.fillna(mean_values)
    train_df = train_df.fillna(0)
    test_df = test_df.fillna(0)

    scaler = MinMaxScaler()
    scaler.fit(train_df.iloc[:, 1:-1])
    serialized_scaler = pickle.dumps(scaler)

    local_model_path = "scalar.pkl"
    edt = pytz.timezone('US/Eastern')
    current_time_edt = datetime.now(edt)
    version = current_time_edt.strftime('%Y%m%d_%H%M%S')
    gcs_model_path = f"{MODEL_DIR}/scalar_{version}.pkl"
    ##
    joblib.dump(serialized_scaler, local_model_path)

    # Upload the model to GCS
    with fs.open(gcs_model_path, 'wb') as f:
        joblib.dump(model, f)
    ##
    train_df.iloc[:, 1:-1] = scaler.transform(train_df.iloc[:, 1:-1])
    test_df.iloc[:, 1:-1] = scaler.transform(test_df.iloc[:, 1:-1])

    train_serialized_data = pickle.dumps(train_df)
    test_serialized_data = pickle.dumps(test_df)

    return train_df,test_df

def training(train_df,test_df):


    #from tensorflow.keras.preprocessing.sequence import pad_sequences
    #import tensorflow as tf

    def pad_trunc(data, max_len):
        new_data = []
        zero_vec = []
        for _ in range(len(data[0][0])):
            zero_vec.append(0.0)
    
        for sample in data:
            if len(sample) > max_len:  # truncate
                temp = sample[:max_len]
            elif len(sample) < max_len:
                temp = sample
                number_additional_elements = max_len - len(sample)
                for _ in range(number_additional_elements):
                    temp.insert(0, zero_vec)  # insert zero vectors at the beginning
            else:
                temp = sample
    
            new_data.append(temp)
    
        return new_data

    def create_sequence_tr_test_data(df, feature_columns, y_label='y'):
        id_list = df['id'].unique()
        x_sequence = []
        y_local = []
        ids = []
        for idx in id_list:
            n_row_idx = df[df['id'] == idx].shape[0]
            if n_row_idx >= 15:
                ids.append(idx)
                x_sequence.append(df[df['id'] == idx][feature_columns].iloc[:, 1:].values.tolist())
                y_local.append(df[df['id'] == idx]['SepsisLabel'].iloc[:])        
        return x_sequence, y_local, ids
    
    feature_columns = test_df.iloc[:, :-1].columns

    X_train, y_train, id_ls_train = create_sequence_tr_test_data(train_df, feature_columns)
    X_test, y_test, id_ls_test = create_sequence_tr_test_data(test_df, feature_columns)

    batch_size = 32
    embedding_dim = 39
    max_len = 50
    embedding_dim = 39

    X_train = pad_trunc(X_train, max_len)
    X_test = pad_trunc(X_test, max_len)

    y_train = pad_sequences(y_train, padding='post', dtype='float32')
    y_test = pad_sequences(y_test, padding='post', dtype='float32')  

    X_train = np.reshape(X_train, (len(X_train), max_len, embedding_dim))
    X_test = np.reshape(X_test, (len(X_test), max_len, embedding_dim))
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train,X_test,y_train,y_test


def model(X_train,X_test,y_train,y_test):


    #from tensorflow.keras.preprocessing.sequence import pad_sequences
    #import tensorflow as tf

    if __name__ == "_main_":
        #warnings.filterwarnings("ignore")

    
        max_len = 50
        embedding_dim = 39
        num_neurons = 50
        epochs = 60
        batch_size = 64
        drop_rate = 0.2
        optimizer = 'RMSprop'

        def F1_score(y_true, y_pred):
            precision = precision_metric(y_true, y_pred)
            recall = recall_metric(y_true, y_pred)
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            return f1

        # Create Precision and Recall objects outside of the function
        precision_metric = tf.keras.metrics.Precision()
        recall_metric = tf.keras.metrics.Recall()
    
    
        model = Sequential()
        model.add(LSTM(num_neurons, input_shape=(max_len, embedding_dim), return_sequences=True))
        model.add(Dropout(drop_rate))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), F1_score])
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))

        # Make predictions
        test_predict = model.predict(X_test)
        train_predict = model.predict(X_train)

        local_model_path = "model.pkl"
        edt = pytz.timezone('US/Eastern')
        current_time_edt = datetime.now(edt)
        version = current_time_edt.strftime('%Y%m%d_%H%M%S')
        gcs_model_path = f"{MODEL_DIR}/model_{version}.pkl"


        model.save('saved_model')
        joblib.dump(model, local_model_path)

        # Upload the model to GCS
        with fs.open(gcs_model_path, 'wb') as f:
            joblib.dump(model, f)
 
def main():
    """
    Main function to orchestrate the loading of data, training of the model,
    and uploading the model to Google Cloud Storage.
    """
    # Load and transform data
    gcs_train_data_path = "gs://sepsis_pred_bucket/data/train/train.csv"
    train_df = load_train_data(gcs_train_data_path)
    gcs_test_data_path = "gs://sepsis_pred_bucket/data/train/test.csv"
    test_df = load_test_data(gcs_test_data_path)
    train_df,test_df=feature_engineering(train_df,test_df)
    train_df,test_df=mean_imputation(train_df,test_df)
    X_train,X_test,y_train,y_test=training(train_df,test_df)
    model(X_train,X_test,y_train,y_test)

    
    

if __name__ == "_main_":
    main()