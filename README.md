# Salary Prediction
Repository holds end-to-end regression project, with solution for deployment using PostgreSQL database and Airflow. 

### Problem definition
An HR company wants to provide a professional advice on salary for its customers. It turns out to be very a valuable information for   both the candidates, that are not sure how much money they should aim for in negotiation process and for companies that don't want to   offer too low salary for an open position, which could discaurage talented candidates, but on the other hand they don't want to overpay  them, as unnecessarily high salaries mean smaller budget for other companies expenses.  

The company is currently using a simple model, which is average salary per industry and job type, degree and industry, but is looking   for a more accurate solution, as suggesting too low or too high salary for a job position is leading to either not satisfied employees   leaving their employer or not satisfied companies, that eventually finds out that they are paying too much. Result of both cases is   negative review of the HR company, which logically leads to less customers.  

The final product is expected to be a machine learning model delivered in a form, in which it can be both easily used to predict new   salaries but also can be easily maintained and retrained as the labour market changes and new data becomes available.  

### Dataset
Dataset - __data.csv__ has 1 000 000 unique rows with 9 columns. Salary is in thousands dollars per year.

![image](https://user-images.githubusercontent.com/31499140/79072827-788abc80-7ce3-11ea-8f89-3ce8229b1d8f.png)

### Exploration Data Analysis
__Salary_predictions-EDA.ipynb__ contains data exploration. Findings and plots are part of the project presetion. Here is also established a base line, which is the mean salary by job type, degree and industry with  
mean squared error of 743K and  
mean absolute error of 22K.

The new model needs to take into account also miles from metropolis and years of experience.  
Even though relationship between each of this numerical features and salary is linear,   
I chose to go for algorithms that can model also non linear relationships such as tree-based models: Decision Trees, Random Forest and Gradient Boosted Trees.  
The challenge is not in feature selection in this case, but in generating new features, to help the algorithms to see the patterns among categories.    
For this I used summary statistics of each group, where groups will be created by jobType, degree and industry.  
CompanyID was dropped as it doesn't imply and relationship with the target variable, and the model will be thisway generalized towards any company.  


### Regression
The modeling part of the project is happening in __Salary_predictions-MODELING.ipynb__. The dataset notebook is pre-set to work with the __data.csv__, but for production there is a commented out part in cell with input parameters with details to download and save the data to a Postgresql database alongside with the model, categorical variables encoders, selected features and generated features.  

### Deployment
Nootebook __Salary_predictions-PREDICTING.ipynb__ is for predicting salaries of new jobs (if there are any) based on the train model.
Output of this notebook is table with predicted salaries. As it is meant for production where new jobs will be added on regular bases,
only jobs without predicted salary yet are scored and the ones that already have salaries predicted are not touched.  


![image](https://user-images.githubusercontent.com/31499140/79073193-76c1f880-7ce5-11ea-9403-7fa924063199.png)


It is pre-set to work with __unseen_data.csv__, but there is again a commented out part of the code for downloading data from a PostgreSQL database.

Alternatively it is possible to run __Salary_predictions-PREDICTING.py__ from the command line, passing path to the data.
If the file is in the current folder run it witht he command:
__python Salary_predictions-PREDICTING.py unseen_data.csv

File __salary_predictions_DAG.py__ is for the deployment with Airflow server. It has task to execute the notebook Salary_predictions-PREDICTING.ipynb, that is scheduled to run every night.

File __salary_predictions_helper.py__ contains all the classes and functions used in all three stages, EDA, modeling and predicting.

The notebook has a snippet of code, that checks whether there are any new jobs with missing salary at all and stop the execution of the code, if there are not any.

### Presentation 
Project presentation is in the attached powerpoint presentation __Salary_predictions-PRESENTATION.pptx__, or you can view it as a markdown file __Salary_predictions-PRESENTATION.md__ 




