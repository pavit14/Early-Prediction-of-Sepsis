# Early Prediction of Sepsis

The Early Sepsis Prediction Project aims to facilitate the deployment and management of machine learning models for the early prediction of sepsis in clinical settings. This project integrates machine learning best practices with operational excellence to enable efficient collaboration between data scientists, healthcare professionals, and operations teams.


# Description

The aim of this project is to develop a machine learning model to predict sepsis among patients early (at least before 6 hours of occurance) on based on their clinical data.

The data used for this project is from the 2019 PhysioNet Computing in Cardiology Challenge: https://physionet.org/content/challenge-2019/1.0.0/. 

Data:
Dataset consists of 2 parts: training set A (20,336 subjects) and B (20,000 subjects). The data repository contains one file per subject (e.g., training/p00101.psv for the training data). Data is in psv format. Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time). Readings are taken typically once in hour. Each row's last column represent the sepsis label either 0 or 1. Lables in datasets are shifted by 6 hours. 


# Getting Started
# Prerequisites
Before using the Sepsis Prediction MLops framework, ensure you have the following prerequisites installed:
- [Docker](https://www.docker.com/get-started/)
- [Airflow](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
- [DVC](https://airflow.apache.org/docs/apache-airflow/stable/start.html)
- [Python](https://www.python.org/downloads/) 3.x
- Pip (Python package installer)

# Installation 

1) create an virtual environment and clone the repository
```git clone https://github.com/pavit14/Early-Prediction-of-Sepsis.git```

2) Install dependencies 
```pip install -r requirements.txt```

3) Go inside the dags/data folder. Run below bash command to download the dataset. (This command will only run for bash. It will download the dataset directly in the data directory.)
```cd dags/data```
```wget -r -N -c -np --accept="*.psv" https://physionet.org/files/challenge-2019/1.0.0/```

5) Navigate to the dags directory.
```cd ../```

5) Fetch the docker-compose.yaml file. (This file is alrday present in repository with all the necessary changes required to run the airflow. However, the error may occur if there is airflow version difference. We are using airflow version 2.5.1 for this project. If you are using updated version 2.7.1 make sure you have installed docker-compose.yaml file with same verison. Run below command and update the below changes in the file.)
for version 2.7.1: 
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.1/docker-compose.yaml' 

Set below parameters as specified in docker-compose.yaml file:

AIRFLOW__CORE__LOAD_EXAMPLES: 'false'

_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas }

6) Run the below command for setting up the airflow and docker. 
(Ensure that you have installed the docker desktop and check for adequate memory)
 ```docker-compose up --build```


# Workflow
# 1) Data Preprocessing:
Data preprocessing have below steps:
- i) load_train_test_valid_files
- ii) feature_engineering
- iii) preprocess_zero_imput_norm
- iv) preprocess_mean_input_norm

We are using training_set_A as our training dataset and training_set_B is splitted into test and valid dataset. 
For preprocessing, we are doing following steps
Converting .psv files into df
For each patient file, there are many null values as every test is not getting conducted every hour. So, we are assuming that between 2 tests values are remaining same. So, we are using ffill to remove those nulls. 
We are adding the ID column for each patient identifcation. 
We are stacking the all patients' data one below the other and craeted a single dataframe for training, testing and valid each.
However, there are some nulls remaining as there is no record before first record in each column for each patient comes in. So, we are creating 2 types of dataframes, in one we are imputing nulls with zero and in other approach we are imputing it with mean value. 
At last, we are normalizing the data. 

a) First run the below file:(This file includes the preprocessing functions)
Early-Prediction-of-Sepsis/dags/src/pylab1.py

b) Nevigate to 'Early-Prediction-of-Sepsis' using you rcommand line to run this file as this file uses current directory path .
Run the below file to create .csv files to save created dataframes after preprocessing. 
Early-Prediction-of-Sepsis\dags\create_csv.py

c) Now nevigate to dags directory using command line and initialize the database.
docker compose up airflow-init

d) Run the airflow:
docker compose up

e) Visit the 'localhost:8080' and login with credential.
Run the below file to create the run the pipeline dag. 
Early-Prediction-of-Sepsis/dags/airflow.py

f) Once the dataframes saved in csv files, give dvc and git push.

# License
This project is licensed under the https://physionet.org/content/challenge-2019/1.0.0/LICENSE.txt licence. 



















