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



def split_data(df: pd.DataFrame):

    """
    Split the dataframe into train and test set
    Args:
        pd.DataFrame: Pandas dataframe
    Returns:
        X_train: Features train set
        y_train: Target train set 
        X_test: Features test set
        y_test: Target test set
    """

    X = df.drop("Attrition",axis=1)
    y= df["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.30,
                                                    random_state=42,
                                                    stratify=y)  

    X_train.to_csv(r'../data/X_train.csv')
    y_train.to_csv(r'../data/y_train.csv')
    X_test.to_csv(r'../data/X_test.csv')
    y_test.to_csv(r'../data/y_test.csv')
    # return X_train, X_test, y_train, y_test










if __name__ == "__main__":

    df = csv_to_df()

    df.to_csv(r'../data/prepared_df.csv')
    split_data(df)
    

    print("Script running fine") 