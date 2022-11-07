import pickle
import random


import numpy as np
import pandas as pd

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split


from prefect import flow, task
from prefect.task_runners import SequentialTaskRunner

import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_tracking_uri("sqlite:///mydb.sqlite")
EXPERIMENT_NAME = "IBM_Employee_Attrition_Prediciton"
mlflow.set_experiment(EXPERIMENT_NAME)


@task(name="CSV_to_Dataframe", retries=3)
def csv_to_df() -> pd.DataFrame:

    """
    Function to extract data from the csv file and returns a pandas dataframe
    Returns:
        pd.DataFrame: A pandas dataframe

    """

    df = pd.read_csv("/Users/DEEPANSHU/Folder001/Git_Projects/MLOps/IBM_Employee_Attrition_Prediciton/data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    # print(df.shape)
    # Drop the unnecessary columns

    df.drop(['OverTime','Over18','EmployeeCount', 'EmployeeNumber', 'StandardHours'], axis="columns", inplace=True)
    
    # print("After dropping")
    # print(df.shape)

    return df


@task(name="Train_Test_Spliting ", retries=3)
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

    # X_train.to_csv(r'../data/X_train.csv')
    # y_train.to_csv(r'../data/y_train.csv')
    # X_test.to_csv(r'../data/X_test.csv')
    # y_test.to_csv(r'../data/y_test.csv')
    return X_train, X_test, y_train, y_test


@flow(name="Training_Pipeline ML Model")#,task_runner=SequentialTaskRunner())
def train_pipeline():
    """
    Pipeline to transform data and train the model
    Returns:
        None 
    """

    df = csv_to_df()
    X_train, X_test, y_train, y_test = split_data(df)

    ohe_encoder = OneHotEncoder()
    ord_encoder = OrdinalEncoder()

    ct = make_column_transformer( 
            (ord_encoder,["BusinessTravel"]),
            (ohe_encoder, ["Department","EducationField", "Gender","JobRole", "MaritalStatus"]),
            remainder = 'passthrough')
    
    pipe = Pipeline([
    ('Column_Transformations', ct),
    ('Logistic Regression', LogisticRegression()),
    ])


    with mlflow.start_run():
        
        mlflow.set_tag("Author", "Deepanshu Kaushik")
        mlflow.set_tag("Model", "Logistic_Regression")
        


        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy_Score = accuracy_score(y_test, y_pred)
        print(accuracy_Score)
        y_pred = pipe.predict_proba(X_test)[:,1]
        ROC_AUC_Score = roc_auc_score(y_test, y_pred)
        print(ROC_AUC_Score)
        
        

        mlflow.log_metric("Accuracy", accuracy_Score)
        mlflow.log_metric("ROC_AUC_score", ROC_AUC_Score)
        mlflow.log_artifact(local_path="models", artifact_path="models/pipe")


        mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="models/logreg_pipe",
            registered_model_name="scikit-learn-Logistic_Regression",
        )

    return pipe











if __name__ == "__main__":

    trained_pipe = train_pipeline()

    with open("models/pipe.pkl", "wb") as f:
        pickle.dump(trained_pipe, f)
    print("Completed: Model pipeline trained and saved")