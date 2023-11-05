import os
import pandas as pd

def load_data():
    data_folder = os.path.join(os.path.dirname(__file__), "../data/training_setA")
    
    # Initialize an empty list to store dataframes
    dataframes = []

    # Iterate through the files in the data folder
    for filename in os.listdir(data_folder):
        if filename.endswith('.psv'):
            # Read each .psv file into a DataFrame
            data = pd.read_csv(os.path.join(data_folder, filename), delimiter='|')
            # Create a 'patientid' column with the filename (excluding the extension)
            data['patientid'] = os.path.splitext(filename)[0]
            # Append the DataFrame to the list
            dataframes.append(data)

    # Concatenate all DataFrames into one
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Ensure temporal order of records for each patient
    combined_data.sort_values(by=['patientid', 'ICULOS'], inplace=True)

    # You can return the combined_data DataFrame if needed
    return combined_data

if __name__ == "__main__":
    # You can run this script as a standalone script if needed
    result = load_data()
    # You can also save the result to a file or perform other operations here
