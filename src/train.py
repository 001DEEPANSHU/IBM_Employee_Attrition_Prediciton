import pickle
import random
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
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