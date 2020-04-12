# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 12:13:58 2020

@author: ivana
"""

#import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import pickle
import psycopg2
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

### Exploratory Data Analysis

class base_regressor():
    """
    Attributes
    ----------
    cat_cols: array
    means_df : DataFrame
    
    """

    def __init__(self):
        self.cat_cols = ['jobType', 'degree', 'industry']
        self.means_df = None

    def fit(self, X, y, sample_weight=None):
        """
        Fit the regressor.
        
        Functions stores mean of the target variable for each combination of jobType, degree and industry.
        
        Parameters
        ----------
        X : DataFrame
        y : pandas.core.series.Series
        
        Returns
        -------
        self : object
        
        """
        
        if len(y) == 0:
            raise ValueError("y must not be empty.")

        if len(y) != len(X):
            raise ValueError("X and y must have the same length.")
        
        means_df = X.copy()
        means_df['target'] = y
        means_df = means_df[self.cat_cols + ['target']].groupby(self.cat_cols).mean()
        self.means_df = means_df.dropna()        
        return self

    def predict(self, X):
        """
        Adds prediction on test vectors X.
        
        The function merge test dataset with stored means salaries for each combination of jobType, degree and industry. 

        Parameters
        ----------
        X : DataFrame

        Returns
        -------
        y : pd.core.series.Series

        """
        
        if self.means_df is None:
            raise ValueError("The regressor was not fitted yet.")
            
        y = pd.merge(X, self.means_df, how = 'left', left_on = self.cat_cols, right_on = self.cat_cols)
        
        if len(y[y.isnull().any(axis=1)]) > 0:
               print('Salary was not predicted for: ' + str(len(y[y.isnull().any(axis=1)])) + ' cases.')
        else:
            print('For each case was mean salary of given combination of jobType, degree and industry found.')

        return y['target']
    
    
def plot_feature(df, col):
    """
    Makes plot for each feature
    On the left the distribution of samples on the feature.
    On the right the dependance of salary on the feature
    
    Parameters
    ----------
    df : DataFrame
    col : str
    
    """
    
    plt.figure(figsize = (14, 6))
    plt.subplot(1, 2, 1)
    if df[col].dtype == 'int64':
        df[col].value_counts().sort_index().plot()
    else:
        #change the categorical variable to category type and order their level by the mean salary
        #in each category
        mean = df.groupby(col)['salary'].mean()
        df[col] = df[col].astype('category')
        levels = mean.sort_values().index.tolist()
        df[col].cat.reorder_categories(levels, inplace=True)
        df[col].value_counts().plot()
    plt.xticks(rotation=45)
    plt.xlabel(col)
    plt.ylabel('Counts')
    plt.subplot(1, 2, 2)
  
    if df[col].dtype == 'int64' or col == 'companyId':
        #plot the mean salary for each category and fill between the (mean - std, mean + std)
        mean = df.groupby(col)['salary'].mean()
        std = df.groupby(col)['salary'].std()
        mean.plot()
        plt.fill_between(range(len(std.index)), mean.values-std.values, mean.values + std.values, \
                         alpha = 0.1)

    else:
        sns.boxplot(x = col, y = 'salary', data=df)
    
    plt.xticks(rotation=45)
    plt.ylabel('Salaries')
    plt.show()
    
    if df[col].dtype == 'int64':
        sns.scatterplot(df[col], df['salary'])
      

def plot_heatmap(data, numeric_vars, cat_vars):
    """
    Creates heatmap with linear correlations among variables.
    
    Function first takes copy of passed dataframe and encodes categorical variables with average salary for given category.
    
    Parameters
    ----------
    data : DataFrame
    numeric_vars : array
    cat_vars: array
    
    """
        
    df = data.copy()
    for col in df.columns:
        if df[col].dtype.name == "category":
            encode_label(df, col)
    df.head()
    
    # Correlations between selected features and response
    # jobId is discarded because it is unique for individual
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(df[numeric_vars + cat_vars + ['salary']].corr(), cmap='Blues', annot=True)
    plt.xticks(rotation=45)
    plt.show()    
    
    
      
def encode_label(df, col):
    """
    Encodes categorical variables with salary mean.

    Parameters
    ----------
    df : DataFrame
    col : str
     
    """

    #encode the categories using average salary for each category to replace label
    cat_dict ={}
    cats = df[col].cat.categories.tolist()
    for cat in cats:
        cat_dict[cat] = df[df[col] == cat]['salary'].mean()   
    df[col] = df[col].map(cat_dict)    
    
    
def summarize_cat_vars(data, cat_vars):
    """
    Prints summary of categorical variables.
    
    Function first plots number of categorical variables in the dataframe,
    and then prints cardinality and categories for each variables.
    After that prints rare labels in each category.

    Parameters
    ----------
    df : DataFrame
    cat_vars : array
     
    """

    print('Number of categorical variables: ', len(cat_vars))
    print()
    # checking cardinality
    for var in cat_vars:
        print(var, data[var].nunique(), ' categories')
        if data[var].nunique() <= 10:
            print(data[var].unique())
            print()
    print()
    for variable in cat_vars:
        rare_labels = analyse_rare_labels(data, variable, 0.01)
        if len(rare_labels) > 0:
            print('Rare labels in ' + str(variable))
            print(rare_labels)
        else:
            print('There are no rare labels in ' + str(variable))

def analyse_rare_labels(df, var, rare_perc):
    """
    Prints summary of categorical variables.
    
    Function first plots number of categorical variables in the dataframe,
    and then prints cardinality and categories for each variables.

    Parameters
    ----------
    df : DataFrame
    cat_vars : array
     
    """

    df = df.copy()
    tmp = df.groupby(var)['salary'].count() / len(df)
    return tmp[tmp<rare_perc]
  
    
    
def plot_categorical_heatmap(data, col1, col2):
    df_gp = data[[col1,col2,'salary']]
    grouped_df = df_gp.groupby([col1,col2],as_index=False).mean()
    grouped_pivot = grouped_df.pivot(index=col1,columns=col2)
    #fill missing values with 0
    grouped_pivot = grouped_pivot.fillna(0)
    
    fig, ax = plt.subplots()
    im = ax.pcolor(grouped_pivot, cmap='coolwarm')

    #label names
    row_labels = grouped_pivot.columns.levels[1]
    col_labels = grouped_pivot.index

    #move ticks and labels to the center
    ax.set_xticks(np.arange(grouped_pivot.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(grouped_pivot.shape[0]) + 0.5, minor=False)
    
    #insert labels
    ax.set_xticklabels(row_labels, minor=False)
    ax.set_yticklabels(col_labels, minor=False)

    #rotate label if too long
    plt.xticks(rotation=90)

    fig.colorbar(im)
    


### Data Modeling
class Data:
    """
    Attributes
    ----------
    from_database : bool
    database_credentails : DataFrame
    cat_cols : array
    num_cols : array
    feature_cols : array
    target_col : str
    label_encoders : dictionary
    mean_encoder_dict : dictionary
    mean_encoders : dictionary
    train_df : DataFrame
    validate_df : DataFrame
    
    """

    def __init__(self, data_file, cat_cols, num_cols, target_col, database_credentials=None, from_database=False):
        self.from_database = from_database
        self.database_credentails = database_credentials
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.feature_cols = cat_cols + num_cols
        self.target_col = target_col
        self.label_encoders = {}
        self.mean_encoder_dict = {}
        self.mean_encoders = {}
        self.train_df, self.validate_df = self._create_dataframes(data_file)
                
    def label_encode_df(self, df, cols, target_col):
        """
        Creates one label encoder for each ordinal column in the data object instance.
        
        Parameters
        ----------
        df : DataFrame
        cols : array
        traget_col : str

        """
        
        for col in cols:
            if (col in self.label_encoders):
                # if label encoder already exists for col, it will be used
                self._label_encode(df, col, label_enc = self.label_encoders[col])
            elif (col in self.mean_encoders):
                # if label encoder already exists for col, it will be used
                self._label_encode(df, col, label_enc = self.mean_encoder_dict[col])
            else:
                self._label_encode(df, col, target_col)

    def update_feature_cols(self, features):
        """
        Updates features_cols parameter with array of features to use in modeling.
        
        Parameters
        ----------
        features : array

        """
        
        self.feature_cols = features
 
    def _label_encode(self, df, col, target_col=None, label_enc=None):
        """
        Label encodes the data and saves the encoder.

        In case of ordinal categories is proper order used, and in case of nominal categories is used frequency of that category in the dataset.
        Function first check whether label_enc was passed, if yes used proper inputs for label_enc.transform in case of ordinal categorical variables
        and incase of the rest of categorical variables, where encoder replaces category with its average target variable.
        If there is no label_enc already, an OrdinalEncoder is created for ordinal categorical variables and TargetMeanEncoder in case of the
        categorical variables. Dictionary of TargetMeanEncoder for given category is separately stored, so it could be saved afterwards.
        
        Parameters
        ----------
        df : DataFrame
        col : array
        target_col : str
        label_enc : dictionary

        """

        if label_enc:
            if col in ['jobType', 'degree']:
                df[col] = label_enc.transform(np.array(df[col]).reshape(-1,1))
            else:
                df[col] = label_enc.transform(df[[col]], col)
        else:
            if col in ['degree', 'jobType']:
                if col == 'degree':
                    categories_ordered  = [['NONE', 'HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL']]
                elif col == 'jobType':
                    categories_ordered = [['JANITOR', 'JUNIOR', 'SENIOR', 'MANAGER', 'VICE_PRESIDENT', 'CTO', 'CFO', 'CEO']]

                ordinal_enc = OrdinalEncoder(categories = categories_ordered)
                ordinal_enc.fit(np.array(df[col]).reshape(-1,1))
                df[col] = ordinal_enc.transform(np.array(df[col]).reshape(-1,1))
                self.label_encoders[col] = ordinal_enc

            else:
                target_mean_enc = TargetMeanEncoder()
                target_mean_enc.fit(df[[col]], col, df[self.target_col])
                df[col] = target_mean_enc.transform(df[[col]], col)
                self.mean_encoder = target_mean_enc
                self.mean_encoder_dict[col] = target_mean_enc
                self.mean_encoders[col] = target_mean_enc.get_encoder()
    
    def _create_dataframes(self, data_file):
        """
        Loads and merges training data features and targets, preprocesses data and encodes data.
        
        Both train_df and validate_df are created, and in case of validate_df is encoding and preprocessing
        first fit on train_df, as this part of dataframe is for model validation.
        
        Parameters
        ----------
        train_feature_df : array
        train_target_df : str
        
        Returns
        ----------
        train_df : DataFrame
        validate_df : DataFrame

        """
        data = self._load_data(data_file)
        train_feature_df = data.drop(self.target_col, axis = 1)
        train_target_df = data[['jobId', self.target_col]].copy()

        X_train, X_validate, y_train, y_validate = train_test_split(train_feature_df, train_target_df, 
                                                    test_size=0.3, 
                                                    random_state=1)
                                                                    
        train_df = self._merge_dfs(X_train, y_train)
        validate_df = self._merge_dfs(X_validate, y_validate)
        #fitting and preprocessing train data                                                            
        train_df = self._clean_data(train_df)
        self.label_encode_df(train_df, self.cat_cols, self.target_col)
        #preprocessing validation data
        validate_df = self._clean_data(validate_df)
        self.label_encode_df(validate_df, self.cat_cols, self.target_col)
        self.mean_encoder_df = self.mean_encoder.get_encoder()
        print('Label encoded train dataframe:')
        try:
            display(train_df.head())
        except:
            print(train_df.head())
        print()
        print('Label encoded validate dataframe:')
        try:
            display(validate_df.head())
        except:
            print(validate_df.head())
        return train_df, validate_df
        
    def _load_data(self, file, from_database=None, database_credentails=None):
        """
        Loads the data.
        
        Function checks whether parameter from_database = False, if yes, the data are read from a csv file,
        otherwise are data loaded from a PostgreSQL database.
           
        Parameters
        ----------
        file : str
        from_database : Bool
        
        Returns
        ----------
        df : DataFrame

        """
        if self.from_database == False:
            df = pd.read_csv(file)
        
        else:
            conn = psycopg2.connect(host = self.database_credentials['host'].item(),
                                    database = self.database_credentials['database'].item(), 
                                    user = self.database_credentials['user'].item(), 
                                    password = self.database_credentials['password'].item())

            # creates a new cursor used to execute SELECT statements
            cur = conn.cursor()
            # creating the query 
            postgreSQL_select_Query = self.sql_query
            # quering the data
            cur.execute(postgreSQL_select_Query)
            data = cur.fetchall() 

            # puting data into a dataframe
            data = pd.DataFrame.from_records(data, columns = ['jobId', 'companyId', 'jobType', 'degree', 'major', 'industry', 'salary'])
            
            cur.close()
            conn.close()
            
        return df
    
    def _merge_dfs(self, df1, df2, key=None, left_index=False, right_index=False):
        return pd.merge(left=df1, right=df2, how='inner', on=key, left_index=left_index, right_index=right_index)
    
    def _clean_data(self, df):
        """
        Remove rows that contain salary <= 0 or duplicate job IDs.
        
        Parameters
        ----------
        df : DataFrame
        
        Returns
        ----------
        df : DataFrame

        """

        df = df.drop_duplicates(subset='jobId')
        df = df[df.salary>0]
        return df

class TargetMeanEncoder():
    
    """
    Encoder for nominal categorical variables
    
    Each category is labeled by mean of target variable. 
    """

    def __init__(self):
        self.encoder_dict_ = {}

    def fit(self, df, col, y):
        """
        Groups dataframe by passed column and calculates mean target for each group.
        
        Function loops through each category and saves mean value for it in a dictionary.
        
        Parameters
        ----------
        df : DataFrame
        col : str
        y : pandas.Series
        
        """
        
        df['target'] = y
        df = df.groupby(col).mean().reset_index()

        for category in df[col].unique():
            self.encoder_dict_[category] = df.loc[df[col] == category, 'target']
        return self

    def transform(self, df, col):
        """
        Loops through categories in passed column and replace it by mean value of salary found in a dictionary
        for given category.

        Parameters
        ----------
        df : DataFrame
        col : str
      
        Returns
        -------
        df['encoded'] : pandas.Series
        
        """
        
        for category in df[col].unique():
            df.loc[df[col] == category, 'encoded'] = self.encoder_dict_[category].item()
        return df['encoded']
    
    def get_encoder(self):
        """
        Retrieves dictionary with mean value for each category.
        
        """
        return self.encoder_dict_
    
class FeatureGenerator:
    """
    Initializes class and creates groupby object for data.

    Attributes
    ----------
    data : Data object
    cat_cols : array
    groups : DataFrame
    groups_stats_df : DataFrame

    """
        
    def __init__(self, data):
        
        self.data = data
        self.cat_cols = data.cat_cols
        self.groups = data.train_df.groupby(self.cat_cols)
        self.group_stats_df = None
        
    def add_group_stats(self):
        """
        Adds group statistics to data stored in data object.

        """
            
        #get group stats
        self.group_stats_df = self._get_group_stats()
        group_stats_df = self.group_stats_df
        group_stats_df.reset_index(inplace=True)
  
        #merge derived columns to original df
        self.data.train_df = self._merge_new_cols(self.data.train_df, group_stats_df, self.cat_cols, fillna=True)
        self.data.validate_df = self._merge_new_cols(self.data.validate_df, group_stats_df, self.cat_cols, fillna=True)      
        print('Train dataframe with group stats:')
        try:
            display(self.data.train_df.head())
        except:
            print(self.data.train_df.head())
        print()
        print('Validate dataframe with group stats:')
        try:
            display(self.data.validate_df.head())
        except:
            print(self.data.validate_df.head())
        
        #update column lists
        group_stats_cols = ['group_mean', 'group_max', 'group_min', 'group_std', 'group_median']
        self._extend_col_lists(self.data, cat_cols=group_stats_cols)  
        
    def _get_group_stats(self):
        """
        Calculates group statistics.
        
        Returns
        ----------
        group_stats_df : DataFrame

        """
        
        target_col = self.data.target_col
        group_stats_df = pd.DataFrame({'group_mean': self.groups[target_col].mean()})
        group_stats_df['group_max'] = self.groups[target_col].max()
        group_stats_df['group_min'] = self.groups[target_col].min()
        group_stats_df['group_std'] = self.groups[target_col].std()
        group_stats_df['group_median'] = self.groups[target_col].median()
        return group_stats_df
        
    def _merge_new_cols(self, df, new_cols_df, keys, fillna=False):
        """
        Removes rows that contain salary <= 0 or duplicate job IDs.
        
        Parameters
        ----------
        df : DataFrame
        new_cols_df : DataFrame
        keys : array
        fillna : Bool
        
        Returns
        ----------
        df : DataFrame

        """
        
        df = pd.merge(df, new_cols_df, on=keys, how='left')
        if fillna:
            df.fillna(0, inplace=True)
        return df
        
    def _extend_col_lists(self, data, cat_cols=[], num_cols=[]):
        """
        Adds engineered feature cols to data col lists.
        
        Parameters
        ----------
        cat_cols : array
        num_cols : array

        """

        data.num_cols.extend(num_cols)
        data.cat_cols.extend(cat_cols)
        data.feature_cols.extend(num_cols + cat_cols)
        
class ModelContainer:
    """
    Attributes
    ----------
    models : array
    best_model : Obj
    predictions : array
    mean_mse : dictionary
    hyper_parameters : dictionary

    """
    def __init__(self):

            
        self.models = []
        self.best_model = None
        self.predictions = None
        self.mean_mse = {}
        self.hyper_parameters = {}
        #self.default_num_iters = default_num_iters
        #self.verbose_lvl = verbose_lvl
        
    def add_model(self, model, hyper_parameters):
        """
        Adds model and it's hyper parameters.
        
        Parameters
        ----------
        model : Obj
        hyper_parameters : dictionary
        
        """
            
        self.models.append(model)
        self.hyper_parameters[model] = hyper_parameters

    def cross_validate(self, data, k=5, num_procs=1):
        """
        Cross validates models using given data.
        
        The results of models are stored in mean_mse dictionary.
        
        Parameters
        ----------
        data : Data object
        k : int
        num_procs : int

        """
            
        feature_df = data.train_df[data.feature_cols]
        target_df = data.train_df[data.target_col]
        for model in self.models:
            neg_mse = cross_val_score(model, feature_df, target_df, cv=k, n_jobs=num_procs, scoring='neg_mean_squared_error')
            self.mean_mse[model] = -1.0*np.mean(neg_mse)
            
    def validate_best_model(self, data, num_procs=1):
        """
        Validates the best model on validation dataframe.
        
        Parameters
        ----------
        data : Data object
        num_procs : int
        
        """
        feature_df = data.validate_df[data.feature_cols]
        target_df = data.validate_df[data.target_col]
        self.best_model_predict(feature_df)
        mse = mean_squared_error(target_df, self.predictions)
        mae = mean_absolute_error(target_df, self.predictions)        
        print('Best model MSE on validation test is: ', mse)
        print('Best model MAE on validation test is: ', mae)
    
    def tune_best_model(self,data):
        """
        Uses grid search to find the best hyper parameters for the best model.
        
        Parameters
        ----------
        data : Data object

        """
            
        print(self.hyper_parameters[self.best_model])
        self.best_model = GridSearchCV(estimator = self.best_model,
                                   param_grid = self.hyper_parameters[self.best_model],
                               #    scoring = 'neg_mean_squared_error',
                                   cv = 5)
        feature_df = data.train_df[data.feature_cols]
        target_df = data.train_df[data.target_col]
        self.best_model_fit(feature_df, target_df)
    
    def select_best_model(self):
        """
        Selects the best model based on smallest mean squared error.

        """
            
        self.best_model = min(self.mean_mse, key=self.mean_mse.get)
        
    def best_model_fit(self, features, target):
        """
        Fits the best model to the data.
        
        Parameters
        ----------
        featres : array
        target : str

        """
        self.best_model.fit(features, target)
    
    def best_model_predict(self, features):
        """
        Remove rows that contain salary <= 0 or duplicate job IDs.
        
        Parameters
        ----------
        features : array

        """
        self.predictions = self.best_model.predict(features)
        
    @staticmethod
    def get_feature_importance(model, cols):
        """
        Retrieves and sorts feature importancess.
        
        Parameters
        ----------
        model : Obj
        cols : array
        
        Returns
        ----------
        feature_importances : DataFrame

        """
            
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_importances = pd.DataFrame({'feature':cols, 'importance':importances})
            feature_importances.sort_values(by='importance', ascending=False, inplace=True)
            #set index to 'feature'
            feature_importances.set_index('feature', inplace=True, drop=True)
            return feature_importances
        else:
            #some models don't have feature_importances_
            return "Feature importances do not exist for given model"

    def print_summary(self):
        """
        Prints summary of models, best model, and feature importance.

        """
            
        print('\nModel Summaries:\n')
        for model in self.mean_mse:
            print('\n', model, '- MSE:', self.mean_mse[model])
        print('\nBest Model:\n', self.best_model)
        print('\nMSE of Best Model\n', self.mean_mse[self.best_model])
        print('\nFeature Importances\n', self.get_feature_importance(self.best_model, data.feature_cols))

        feature_importances = self.get_feature_importance(self.best_model, data.feature_cols)
        feature_importances.plot.bar()
        plt.show()
        
def save_model(data, feature_generator, models_tuned):
    """
    Saves label encoders, list of selected features, generated features and the tuned model
    
    The functions saves objects either to the disk or to the database, based on value of data.from_database parameter.
    In case of saving objects to the database, it is assumed that the table models with requried columns already exists.
    
    Parameters:
    ----------
    data : Data object
    feature_generator : FeatureGenerator object
    models_tuned : ModelContainer object
    """
    
    if data.from_database:
               
                # saving label encoders, generated features, selected features and the model to the disk                
                conn = psycopg2.connect(host = data.database_credentials['host'].item(),
                                        database = data.database_credentials['database'].item(), 
                                        user = data.database_credentials['user'].item(), 
                                        password = data.database_credentials['password'].item())
                cur = conn.cursor()

                # Assuming thre is postgres table 'models' with columns model_name, model_file, label_encoder_file, mean_encoder_file, selected_features and generated_features
                pickled_model = pickle.dumps(models_tuned.best_model) 
                pickled_label_encoders = pickle.dumps(data.label_encoders)
                pickled_mean_encoders = pickle.dumps(data.mean_encoders)
                pickled_selected_features = pickle.dumps(data.feature_cols)
                pickled_generated_features = pickle.dumps(feature_generator.group_stats_df)
                sql = "INSERT INTO models (model_name, model_file, label_encoder_file, mean_encoder_file, selected_features, generated_features)  VALUES(%s, %s, %s, %s, %s, %s )"
                cur.execute(sql, ('salary_prediction', psycopg2.Binary(pickled_model), psycopg2.Binary(pickled_label_encoders), psycopg2.Binary(pickled_mean_encoders), 
                                  psycopg2.Binary(pickled_selected_features), psycopg2.Binary(pickled_generated_features)))
                conn.commit() 
                
                cur.close()
                conn.close()
    else:
        # save label encoders, generated features, selected features and the model to the disk
        pickle.dump(data.label_encoders, open('label_encoders.pkl', 'wb')) 
        pickle.dump(data.mean_encoders, open('mean_encoders.pkl', 'wb')) 
        pickle.dump(data.feature_cols, open('feature_cols.pkl', 'wb')) 
        pickle.dump(feature_generator.group_stats_df, open('group_stats_df.pkl', 'wb')) 
        pickle.dump(models_tuned.best_model, open('model_gb.pkl', 'wb')) 
    
### Data Predicting
class NewData:
    """
    Attributes
    ----------
    cat_cols : array
    num_cols : array
    from_database : Boolean
    new_file : str
    feature_cols : array
    target_col : str
    label_encoders : dictionary
    mean_encoders : dictionary
    group_stats_df : DataFrame
    model : Obj
    new_data = DataFrame
    
    """
    
    def __init__(self, cat_cols, num_cols, from_database, target_col, new_data_file=None, feature_cols=None, label_encoders=None, mean_encoders=None, group_stats_df=None, model=None, database_credentials=None):
        self.cat_cols = list(cat_cols)
        self.num_cols = list(num_cols)
        self.from_database = from_database
        self.new_data_file = new_data_file
        self.feature_cols, self.label_encoders, self.mean_encoders, self.group_stats_df, self.model = self._load_model(from_database, database_credentials, feature_cols, label_encoders, mean_encoders, group_stats_df, model)
        self.target_col = target_col
        self.new_data, self.predicted_data = self._create_dataframes(new_data_file)     
        
    def preprocess(self):
        """
        Label encodes the categorical variables and adds the generated features to the dataframe.
        
        """
            
        self._label_encode_df(self.new_data, self.cat_cols, self.target_col)
        self.new_data = self._add_group_stats(self.new_data)
        try:
            display(self.new_data.head())
        except:
            print(self.new_data.head())
        
        
    def get_predictions(self):
        """
        Adds predictions to the dataframe.
        
        """
            
        self.new_data[self.target_col] = self.model.predict(self.new_data[self.feature_cols])

        
    def save_predictions(self):
        """
        Saves original dataframe alongside with the predictions.

        The functions checks whether paramter from_database = False, if yes the dataframe is saved as a csv file,
        otherwise it replaces the table in Postgresql database.

        """
        
        new_predicted_data, predicted_data = self._create_dataframes(self.new_data_file)
        new_predicted_data.drop(self.target_col, axis = 1, inplace = True)
        new_predicted_data = pd.merge(new_predicted_data, self.new_data[['jobId', self.target_col]], how = 'left', on = 'jobId')
        
        try:
            display(new_predicted_data.head())
        except:
            print(new_predicted_data.head())
             
        new_predicted_data = pd.concat([self.predicted_data, new_predicted_data])

        if self.from_database == False:
            new_predicted_data.to_csv(self.new_data_file, index = False)
            
            return print('Dataframe was saved as a csv file.')
        
        else:
            engine = create_engine(engine_link = self.database_credentials['engine'].item())  
            new_predicted_data.to_sql('predicted_jobs', engine, if_exists = 'replace', index = False, method = 'multi')    
            engine.dispose()
            
            return print('Data in the database were updated.')
        
    def _load_model(self, from_database, database_credentials=None, feature_cols=None, label_encoders=None, mean_encoders=None, group_stats_df=None, model=None):
        """
        Loads selected features, label encoders, mean encoders, group stats dataframe and model either from database or from the disk
        
        Parameters
        ----------
        from_database : Boolean
        feature_cols : str
        label_encoders : str
        mean_encoders : str
        group_stats_df : str
        model : str
        
        Returns
        -------
        feature_cols : Array
        label_encoders : Dictionary
        mean_encoders : Dictionary
        group_stats_df : DataFrame
        model : Object
        
        """

        if from_database:
            conn = psycopg2.connect(host = database_credentials['host'].item(),
                                     database = database_credentials['database'].item(), 
                                     user = database_credentials['user'].item(), 
                                     password = database_credentials['password'].item())

            # creates a new cursor used to execute SELECT statements
            cur = conn.cursor()

            # Loading the selected features
            postgreSQL_select_Query = """
                              select selected_features, label_encoder_file, mean_encoder_file, generated_features, model_file
                              from models
                              where model_name = 'salary_prediction'
                            """

            cur.execute(postgreSQL_select_Query)
            data = cur.fetchone() 
            feature_cols = pickle.loads(data[0][0])
            label_encoders = pickle.loads(data[0][1])        
            mean_encoders = pickle.loads(data[0][2])       
            group_stats_df = pickle.loads(data[0][3])       
            model = pickle.loads(data[0][4])       
            
            cur.close()
            conn.close()
            
        else:
            feature_cols = pickle.load(open(feature_cols, 'rb'))
            label_encoders = pickle.load(open(label_encoders, 'rb'))
            mean_encoders = pickle.load(open(mean_encoders, 'rb'))
            group_stats_df = pickle.load(open(group_stats_df, 'rb'))
            model = pickle.load(open(model, 'rb'))
            
        return feature_cols, label_encoders, mean_encoders, group_stats_df, model

    def _label_encode_df(self, df, cols, target_col):
        """
        Encodes the categorical variables using stored label and mean encoders.
   
        """
        
        for col in cols:
            if col in self.label_encoders:
                # if label encoder already exists for col, it will be used
                self._label_encode(df, col, label_enc = self.label_encoders[col])
            elif col in self.mean_encoders:
                self._label_encode(df, col, mean_enc = self.mean_encoders[col])
            else:
                raise SystemExit("There is no label encoder for variable " + str(col) + "!")

    def _label_encode(self, df, col, target_col=None, label_enc=None, mean_enc=None):
        """
        Label encodes the data based on passed encoders.

        In case of ordinal categories is proper order used, and in case of nominal categories is used mean salary of given category.

        Parameters
        ----------
        df : DataFrame
        col : str
        target_col : str
        label_enc : dictionary
        mean_enc : dictionary
        
        """

        if label_enc:
            df[col] = label_enc.transform(np.array(df[col]).reshape(-1,1))
        else:
            for category in df[col].unique():
                df.loc[df[col] == category, col] = mean_enc[category].item()
    
    def _create_dataframes(self, new_data_file):
        """
        Loads the dataframe.
        
        Parameters
        ----------
        new_data_file : str
        
        Returns
        ----------
        data : DataFrame

        """

        new_data, predicted_data = self._load_data(new_data_file)
        return new_data, predicted_data 
          
    def _load_data(self, file=None):
        """
        Loads the data.
        
        Function checks whether parameter from_database = False, if yes, the data are read from a csv file,
        otherwise are data loaded from a PostgreSQL database.
           
        Parameters
        ----------
        file : str
        from_database : Bool
        
        Returns
        ----------
        df : DataFrame

        """
        if self.from_database == False:
            df = pd.read_csv(file)
        
        else:
            conn = psycopg2.connect(host = self.database_credentials['host'].item(),
                                    database = self.database_credentials['database'].item(), 
                                    user = self.database_credentials['user'].item(), 
                                    password = self.database_credentials['password'].item())

            # creates a new cursor used to execute SELECT statements
            cur = conn.cursor()
            # creating the query 
            postgreSQL_select_Query = self.new_data_file
            # quering the data
            cur.execute(postgreSQL_select_Query)
            data = cur.fetchall() 

            # puting data into a dataframe
            df = pd.DataFrame.from_records(data, columns = ['jobId', 'companyId', 'jobType', 'degree', 'major', 'industry', 'salary'])
            
            cur.close()
            conn.close()
        
        df.head()
        new_data = df[df[self.target_col].isnull()].copy()
        predicted_data = df[df[self.target_col].notnull()].copy()
            
        return new_data, predicted_data
    
    def _add_group_stats(self, data):
        """
        Merges dataframe with the dataframe containing generated features.
        
        Parameters
        ----------
        data : DataFrame
        
        Returns
        ----------
        data : DataFrame

        """
            
        #merge derived columns to original df
        data = self._merge_new_cols(data, self.group_stats_df, self.cat_cols, fillna=True)
        return data 
    
    def _merge_new_cols(self, df, new_cols_df, keys, fillna=False):        
        """
        Merges two dataframes.
        
        Parameters
        ----------
        df : DataFrame
        new_cols_df : DataFrame
        keys : array
        fillna : Bool
        
        Returns
        ----------
        df : DataFrame

        """
            
        df = pd.merge(df, new_cols_df, on=keys, how='left')
        if fillna:
            df.fillna(0, inplace=True)
        return df