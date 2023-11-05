import numpy as np, os, sys
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Converting .psv file to np array
def load_challenge_data(file):
    with open(file, 'r') as f:
        header = f.readline().strip()
        column_names = header.split('|')
        data = np.loadtxt(f, delimiter='|')
    return data

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






