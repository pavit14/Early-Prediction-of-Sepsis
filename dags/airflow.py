# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
#from src.pylab1 import load_train_test_valid_files,feature_engineering,preprocess_zero_imput_norm,preprocess_mean_input_norm
#import tensorflow as tf
from src.demo import load_train_files,load_test_files,feature_engineering,mean_imputation, training, model
from airflow import configuration as conf

# Enable pickle support for XCom, allowing data to be passed between tasks
conf.set('core', 'enable_xcom_pickling', 'True')

# Define default arguments for your DAG
default_args = {
    'owner': 'Pavs',
    'start_date': datetime(2023, 9, 17),
    'retries': 0, # Number of retries in case of task failure
    'retry_delay': timedelta(minutes=5), # Delay before retries
}
# Create a DAG instance named 'your_python_dag' with the defined default arguments
dag = DAG(
    'early_sepsis_prediction_pipeline1',
    default_args=default_args,
    description='Data Pipeline',
    schedule_interval=None,  # Set the schedule interval or use None for manual triggering
    catchup=False,
)

from airflow.operators.bash_operator import BashOperator

install_requirements = BashOperator(
    task_id='install_requirements',
    bash_command='pip install tensorflow',
    dag=dag
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
'''
load_train_test_valid_files_task = PythonOperator(
    task_id='load_train_test_valid_files',
    python_callable=load_train_test_valid_files,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)


feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    #op_kwargs={'output_param': 'output_result'},
    dag=dag
)

preprocess_zero_imput_norm_task = PythonOperator(
    task_id='preprocess_zero_imput_norm',
    python_callable=preprocess_zero_imput_norm,
    provide_context=True,
    dag=dag
)

preprocess_mean_input_norm_task = PythonOperator(
    task_id='preprocess_mean_input_norm',
    python_callable=preprocess_mean_input_norm,
    provide_context=True,
    dag=dag
)

# Set task dependencies
load_train_test_valid_files_task >> feature_engineering_task >> (preprocess_zero_imput_norm_task,preprocess_mean_input_norm_task)
'''

load_train_files_task = PythonOperator(
    task_id='load_train_files',
    python_callable=load_train_files,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)
load_test_files_task = PythonOperator(
    task_id='load_test_files',
    python_callable=load_test_files,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)
feature_engineering_task = PythonOperator(
    task_id='feature_engineering',
    python_callable=feature_engineering,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

mean_imputation_task = PythonOperator(
    task_id='mean_imputation',
    python_callable=mean_imputation,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)

training_task = PythonOperator(
    task_id='training',
    python_callable=training,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)
model_task = PythonOperator(
    task_id='model',
    python_callable=model,
    provide_context=True,
    dag=dag,
    execution_timeout=timedelta(minutes=360),
)


# Set task dependencies
install_requirements>>load_train_files_task >> load_test_files_task >> feature_engineering_task >> mean_imputation_task>> training_task >>model_task

# If this script is run directly, allow command-line interaction with the DAG
if __name__ == "__main__":
    dag.cli()