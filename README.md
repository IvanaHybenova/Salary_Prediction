# Salary Prediction
Repository holds end-to-end regression project, with solution for deployment using PostgreSQL database and Airflow. 

### Problem definition
An HR company wants to provide a professional advice on salary for its customers. It turns out to be very a valuable information for   both the candidates, that are not sure how much money they should aim for in negotiation process and for companies that don't want to   offer too low salary for an open position, which could discaurage talented candidates, but on the other hand they don't want to overpay  them, as unnecessarily high salaries mean smaller budget for other companies expenses.  

The company is currently using a simple model, which is average salary per industry and job type, degree and industry, but is looking   for a more accurate solution, as suggesting too low or too high salary for a job position is leading to either not satisfied employees   leaving their employer or not satisfied companies, that eventually finds out that they are paying too much. Result of both cases is   negative review of the HR company, which logically leads to less customers.  

The final product is expected to be a machine learning model delivered in a form, in which it can be both easily used to predict new   salaries but also can be easily maintained and retrained as the labour market changes and new data becomes available.  

### Dataset
Dataset - __Mall_Customers.csv__ has 200 unique rows with 5 columns

![image](https://user-images.githubusercontent.com/31499140/78266332-f261bf80-7505-11ea-98da-644fbaf9f188.png)

### Clustering
The main part of the project is happening in __Customer_Segmentation.ipynb__. The dataset notebook is pre-set to work with the __Mall_Customers.csv__, but for production there is commented out part in cell with input parameters with details to download and save the data to a Postgresql database alongside with the model and standard scaler.

### Deployment
Nootebook __New_data_segments.ipynb__ is for assigning segments to new customers (if there are any) based on existing clusters. 
Output of this notebook is table with both already segmented customers and new customers labeled accordingly.

![image](https://user-images.githubusercontent.com/31499140/78268942-3b674300-7509-11ea-9910-4d8e051e2479.png)


It is pre-set to work with Mall_Customers-New.csv, but again there is commented out part of the code for downloading data, model and scaler from a PostgreSQL database.

Alternatively it is possible to run __New_data_segments.py__ from the command line, passing path to the data, model and scaler.
If the files are all in the same folder run it witht he command:
__python New_data_segments.py Mall_customers-New.csv model.pkl scaler_mapper.pkl__

File __customer_segments_DAG.py__ is for the deployment with Airflow server. It has task to execute the notebook New_data_segments.ipynb, that is scheduled to run every night.

File __customer_segmentation_helper.py__ contains the 'data' class, which methods are used to do the clustering analysis and functions to assign segment to the new data.

The notebook has a snippet of code, that checks whether there are some new customers at all and stop the execution of the code, if there are not any.

### Presentation 
Project presentation is in the attached powerpoint presentation __Customer_Segmentation.pptx__, or you can view it as a markdown file __Customer_Segmentation.md__ - read accompanying text under each slide for full understanding :)

File __Customer_Segmentation-Result.html__ is part of the presentation.


