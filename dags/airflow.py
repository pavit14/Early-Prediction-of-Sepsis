# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from src.pylab import preprocess_train_data,preprocess_test_val_data,train_test_valid_files

from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'your_name',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}

# Create a DAG instance named 'your_python_dag' with the defined default arguments
dag = DAG(
    'early_sepsis_prediction',
    default_args=default_args,
    description='Data Pipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

# Define PythonOperators for each function
'''
# Task to load data, calls the 'load_data' Python function
load_data_task = PythonOperator(
    task_id='load_data_task',
    python_callable=load_data,
    dag=dag,
)
# Task to perform data preprocessing, depends on 'load_data_task'
data_preprocessing_task = PythonOperator(
    task_id='data_preprocessing_task',
    python_callable=train_test_val_split,
    op_args=[load_data_task.output],
    dag=dag,
)
# Task to build and save a model, depends on 'data_preprocessing_task'
build_save_model_task = PythonOperator(
    task_id='build_save_model_task',
    python_callable=preprocess_train_data,
    op_args=[data_preprocessing_task.output, "model2.sav"],
    provide_context=True,
    dag=dag,
)
# Task to load a model using the 'load_model_elbow' function, depends on 'build_save_model_task'
load_model_task = PythonOperator(
    task_id='load_model_task',
    python_callable=preprocess_test_val_data,
    op_args=["model2.sav", build_save_model_task.output],
    dag=dag,
)
'''
# Define your Airflow tasks

train_test_valid_files_task = PythonOperator(
    task_id='train_test_valid_files',
    python_callable=train_test_valid_files,
    provide_context=True,
    #op_kwargs={'output_param': 'output_result'},
    dag=dag
)


preprocess_train_data_task = PythonOperator(
    task_id='preprocess_train_data',
    python_callable=preprocess_train_data,
    provide_context=True,
    #op_kwargs={'output_param': 'output_result'},
    dag=dag
)

preprocess_test_val_data_task = PythonOperator(
    task_id='preprocess_test_val_data',
    python_callable=preprocess_test_val_data,
    provide_context=True,
    dag=dag
)

# Set task dependencies
train_test_valid_files_task >> preprocess_train_data_task >> preprocess_test_val_data_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()
