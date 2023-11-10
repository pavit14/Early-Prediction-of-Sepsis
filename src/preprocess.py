import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np
from .load import load_data
import warnings
import os

# get parent directory of the file
script_dir = os.path.dirname(__file__)

# get root directory of the file
root_dir = os.path.abspath(os.path.join(script_dir, os.pardir))

# Filter out the specific warning
warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)

def preprocess_data():

    combined_data = load_data()

    # Test
    # print(combined_data.head())

    # Identify patients with sepsis and without
    sepsis_patients = combined_data[combined_data['SepsisLabel'] == 1]['patientid'].unique()
    non_sepsis_patients = [patient for patient in combined_data['patientid'].unique() if patient not in sepsis_patients]

    # Count the number of records for each patient
    patient_record_counts = combined_data['patientid'].value_counts()

    filtered_sepsis_patients = [
        patient for patient in sepsis_patients if patient_record_counts.get(patient, 0) > 30
    ]

    filtered_non_sepsis_patients = [
        patient for patient in non_sepsis_patients if patient_record_counts.get(patient, 0) > 30
    ]

    # Create a column 'last_sepsis_record' to identify the last record with SepsisLabel as 1
    combined_data['last_sepsis_record'] = combined_data.groupby('patientid')['SepsisLabel'].cumsum()

    # Drop patients where max(last_sepsis_record) is < 9
    filter9 = list(combined_data.groupby('patientid').filter(lambda x: x['last_sepsis_record'].max() >= 9)['patientid'].unique())

    # Keep sepsis patients that match both filtration criteria
    common_elements_sepsis_patients = [element for element in filtered_sepsis_patients if element in filter9]

    # Remove records where last_sepsis_record >= 10
    combined_data = combined_data[combined_data['last_sepsis_record'] < 10]

    # Combine sepsis and non-sepsis patients for the data set
    filtered_patients = common_elements_sepsis_patients + filtered_non_sepsis_patients

    # Create a DataFrame containing only records for these patients
    filtered_data = combined_data[combined_data['patientid'].isin(filtered_patients)]

    # Drop last_sepsis_record column
    filtered_data = filtered_data.drop(['last_sepsis_record'], axis=1)

    cleaned_df = pd.DataFrame()

    na_cols_to_exclude = ['EtCO2', 'SaO2', 'AST', 'Alkalinephos', 'Bilirubin_direct', 'Bilirubin_total', 'TroponinI', 'Fibrinogen', 'Unit2']
    cat_cols = ['SepsisLabel', 'ICULOS', 'HospAdmTime', 'Unit1', 'Age', 'Gender', 'patientid']

    # Normalize
    scaler = MinMaxScaler()

    for patient, group in filtered_data.groupby('patientid'):
        # Forward fill missing values
        data = group.transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))

        # Drop columns with > 50% missing values based on previous EDA
        data = data.drop(na_cols_to_exclude, axis=1)

        # Drop categorical columns
        data = data.drop(cat_cols, axis=1)

        # Normalize data (assuming you have a `scaler` object)
        normalized_data = scaler.fit_transform(data)
        scaled_df = pd.DataFrame(data=normalized_data, columns=data.columns)

        # Replace remaining NaN values with a constant (-1)
        scaled_df = scaled_df.fillna(-1)

        # Add Sepsis Label and patientid for further processing and create uniform datasets for each patient
        scaled_df['SepsisLabel'] = group['SepsisLabel'].values
        scaled_df['patientid'] = group['patientid'].values

        # Reduce the number of records for each patient to 30 values from the bottom up
        scaled_df = scaled_df.tail(30)

        cleaned_df = pd.concat([cleaned_df, scaled_df], axis=0)
        
    return cleaned_df

if __name__ == "__main__":
    cleaned_data = preprocess_data()
    cleaned_data.to_csv(os.path.join(root_dir, "data", "preprocessed_dataset.csv"))