import numpy as np
from pandas import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("D:/Code/Python/MachineLearning/Week_3/framingham.csv")
df.head()
df.isnull().sum()
df=df.dropna(how="any", axis=0)
