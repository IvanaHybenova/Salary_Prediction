#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 10:01:15 2020

@author: admin
"""


from airflow import DAG
import datetime as dt
from airflow.operators.bash_operator import BashOperator
from airflow.utils.email import send_email


def notify_email(contextDict, **kwargs):
    """Send custom email alerts."""

    # email title.
    title = "Airflow alert: {task_name} Failed".format(**contextDict)
    
    # email contents
    body = """
    Hi Everyone, <br>
    <br>
    There's been an error in the {task_name} job.<br>
    <br>
    Forever yours,<br>
    Airflow bot <br>
    """.format(**contextDict)

    send_email('example@company.io', title, body) 
    
    
    
default_args = {
        'owner': 'ivana',
        'start_date': dt.datetime(2020, 4, 12),
        'depends_on_past': False,
        'email': ['example@company.io'],
        'email_on_failure': False,
        'retries': 0,
        'retry_delay': dt.timedelta(minutes = 2) }

dag = DAG(
        'assign_segments',
        default_args = default_args,
        description = 'Predict salaries to new jobs.',
        # Continue to run DAG once per day at midnight
        schedule_interval = '0 0 * * *',
        catchup = False)

predict_salaries = BashOperator(task_id='predict_salaries', 
                                                      bash_command='jupyter nbconvert --execute --to html $AIRFLOW_HOME/dags/common/Salary_prediction-PREDICTING.ipynb --no-input',
                                                      on_failure_callback=notify_email,dag=dag)

predict_salaries