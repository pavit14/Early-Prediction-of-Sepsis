import datetime as dt
from airflow import DAG
from airflow.operators.bash_operator import BashOperator

#LOCAL_PREPROCESS_FILE_PATH = '/tmp/data_preprocess.py'
#GITHUB_PREPROCESS_RAW_URL = 'https://github.com/MahavirK1997/Timeseries_Lab/blob/master/src/data_preprocess.py'  # Adjust the path accordingly

LOCAL_TRAIN_FILE_PATH = '/tmp/train.py'
GITHUB_TRAIN_RAW_URL = 'https://github.com/MahavirK1997/Timeseries_Lab/blob/master/src/trainer/train.py'  # Adjust the path accordingly

default_args = {
    'owner': 'Sepsis_project',
    'start_date': dt.datetime(2023, 10, 24),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=5),
}

dag = DAG(
    'model_retraining',
    default_args=default_args,
    description='Model retraining at 9 PM everyday',
    schedule_interval='0 21 * * *',  # Every day at 9 pm
    catchup=False,
)


pull_train_script = BashOperator(
    task_id='pull_train_script',
    bash_command=f'curl -o {LOCAL_TRAIN_FILE_PATH} {GITHUB_TRAIN_RAW_URL}',
    dag=dag,
)


env = {
    'AIP_STORAGE_URI': 'gs://sepsis_pred_bucket/model'
}



run_train_script = BashOperator(
    task_id='run_train_script',
    bash_command=f'python {LOCAL_TRAIN_FILE_PATH}',
    env=env,
    dag=dag,
)

# Setting up dependencies
pull_train_script >> run_train_script

