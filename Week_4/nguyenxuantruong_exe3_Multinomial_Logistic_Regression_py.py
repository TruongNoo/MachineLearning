from sklearn.datasets import fetch_20newsgroups_vectorized
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import sparse
from sklearn.linear_model import LogisticRegression

C=3
n_samples = 20000
X, y = fetch_20newsgroups_vectorized(subset='all', return_X_y=True)
X = X[:n_samples]
y = y[:n_samples]
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0] 

lorg = LogisticRegression(multi_class='multinomial', solver='sag', max_iter=5000)
# and train model by Training Dataset
lorg.fit(X_train, y_train)
# Then Predict the Test data
y_pred = lorg.predict(X_test)
# for accuracy
print("Accuracy=", accuracy_score(y_test, y_pred))
# for confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)