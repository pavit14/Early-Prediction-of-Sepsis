# Early Prediction of Sepsis

The Early Sepsis Prediction Project aims to facilitate the deployment and management of machine learning models for the early prediction of sepsis in clinical settings. This project integrates machine learning best practices with operational excellence to enable efficient collaboration between data scientists, healthcare professionals, and operations teams.


# Description

The aim of this project is to develop a machine learning model to predict sepsis among patients early (at least before 6 hours of occurance) on based on their clinical data.

The data used for this project is from the 2019 PhysioNet Computing in Cardiology Challenge: https://physionet.org/content/challenge-2019/1.0.0/. 

Dataset consists of 2 parts: training set A (20,336 subjects) and B (20,000 subjects). The data repository contains one file per subject (e.g., training/p00101.psv for the training data). Data is in psv format. Each training data file provides a table with measurements over time. Each column of the table provides a sequence of measurements over time (e.g., heart rate over several hours), where the header of the column describes the measurement. Each row of the table provides a collection of measurements at the same time (e.g., heart rate and oxygen level at the same time). Readings are taken typically once in hour. Each row's last column represent the sepsis label either 0 or 1. Lables in datasets are shifted by 6 hours. 

# Getting Started
# Prerequisites
Before using the Sepsis Prediction MLops framework, ensure you have the following prerequisites installed:
Docker
Airflow
DVC
Python 3.x
Pip (Python package installer)
Access to the healthcare dataset


# Installation 

1) craete an virtual environment and clone the repository
git clone https://github.com/pavit14/Early-Prediction-of-Sepsis.git

2) Install dependencies 
pip install -r requirements.txt

3) Go inside the dags/data folder. Run below bash command to download the dataset. (This command will only run for bash. This command will download the daset directly in the data directory.)
cd dags
cd data 
wget -r -N -c -np --accept="*.psv" https://physionet.org/files/challenge-2019/1.0.0/

4) Come out of data directory
cd ..

5) Run the below command for setting up the airflow and docker. 
(Ensure that you have installed the docker desktop)
check the memory 
wget -r -N -c -np --accept="*.psv" https://physionet.org/files/challenge-2019/1.0.0/

Fetch the docker-compose.yaml file. (This file is alrday present in repository with all the necessary changes required to run the airflow. However, the error may occur if there is airflow version difference. We are using airflow version 2.5.1 for this project. If you are using updated version 2.7.1 make sure you have installed docker-compose.yaml file with same verison. Run below command and update the below changes in the file.)
for version 2.7.1: 
curl -LfO 'https://airflow.apache.org/docs/apache-airflow/2.7.1/docker-compose.yaml' 

Set below parameters as specified in docker-compose.yaml file

AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
_PIP_ADDITIONAL_REQUIREMENTS: ${_PIP_ADDITIONAL_REQUIREMENTS:- pandas }


# Workflow













