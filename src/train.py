import pickle
import random


import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


# from prefect import flow, task
# from prefect.task_runners import SequentialTaskRunner

# import mlflow
# from mlflow.tracking import MlflowClient



def csv_to_df() -> pd.DataFrame:

    """
    Function to extract data from the csv file and returns a pandas dataframe
    Returns:
        pd.DataFrame: A pandas dataframe

    """

    df = pd.read_csv("/Users/DEEPANSHU/Folder001/Git_Projects/MLOps/IBM_Employee_Attrition_Prediciton/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    # print(df.shape)
    # Drop the unnecessary columns

    df.drop(['Over18','EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis="columns", inplace=True)
    
    # print("After dropping")
    # print(df.shape)

    return df














if __name__ == "__main__":

    df = csv_to_df()

    df.to_csv(r'../data/prepared_df.csv')

    print("Script running fine") 