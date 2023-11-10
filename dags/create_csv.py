import os
import csv

def create_csv_files(directory_path, file_names):
    for file_name in file_names:
        file_path = os.path.join(directory_path,'dags/data', file_name)
        with open(file_path, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)

directory_path = os.getcwd()
file_names = ['Train_ffill_zero_imput.csv', 'Test_ffill_zero_imput.csv', 'Valid_ffill_zero_imput.csv', 'Train_ffill_mean_imput.csv', 'Test_ffill_mean_imput.csv', 'Valid_ffill_mean_imput.csv']

create_csv_files(directory_path, file_names)