# -*- coding: utf-8 -*-
"""
Created on Sun Apr  5 17:15:48 2020

@author: ivana
"""

import os
from salary_predictions_helper import *
import warnings
warnings.filterwarnings("ignore")
import sys

def main(argv):
    
    # path to the dataset
    new_data_file = sys.argv[1]
    
    #define input files (pulling data from the disk)
    from_database = False
    feature_cols = 'feature_cols.pkl'
    label_encoders = 'label_encoders.pkl'
    mean_encoders = 'mean_encoders.pkl'
    group_stats_df = 'group_stats_df.pkl'
    model = 'model_gb.pkl'
    
    ##define input files (pulling data from the database)
    #database_credentials = pd.DataFrame({
    #    'host': [os.environ['HOST_WAREHOUSE']],
    #    'database': [os.environ['NAME_WAREHOUSE']],
    #    'user': [os.environ['USER_WAREHOUSE']],
    #    'password': [os.environ['PASSWORD_WAREHOUSE']],
    #    'engine': [os.environ['ENGINE_WAREHOUSE']]
    #})
    
    ## SQL query to download jobs table
    #new_data_file =  """
    #               SELECT *
    #               FROM predicted_jobs
    #             """     
    
    #define variables
    cat_cols = ['jobType', 'degree', 'major', 'industry']
    num_cols = ['yearsExperience', 'milesFromMetropolis']
    target_col = 'salary'
    
    #### Load data
    data = NewData(cat_cols, num_cols, from_database, target_col, new_data_file, feature_cols, label_encoders, mean_encoders, group_stats_df, model)
    
    #### Check whether there are new customers
    if len(data.new_data) == 0:
        raise SystemExit("There are no new jobs with salaries to predict!")
    
    #### Preprocess data
    data.preprocess()
     
    #### Get predictions
    data.get_predictions()
    
    ### Save predictions
    data.save_predictions
        
if __name__ == "__main__":
   main(sys.argv[1])
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    