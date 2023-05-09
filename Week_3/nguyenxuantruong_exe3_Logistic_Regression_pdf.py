import numpy as np
from pandas import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv(
    "D:/Code/Python/MachineLearning/Week_3/Admission_Predict.csv")

X = data.iloc[:, 1:8]
y = data.iloc[:, 8]

y_labels = []

for i in y:
    if (i >= 0.75):
        y_labels.append(1)
    if (i < 0.75):
        y_labels.append(0)

y_labels = np.array(y_labels)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_labels, train_size=350, shuffle=False)

Xb = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
logReg = LogisticRegression(max_iter=350)
logReg.fit(Xb, y_train)
print(logReg.coef_)
y_pred = logReg.predict(Xb)
print("Accuracy = ", accuracy_score(y_train, y_pred))
print("Precission = ", precision_score(y_train, y_pred))
print("Recall = ", recall_score(y_train, y_pred))

def logistic_sigmoid_regression(X, y, w_init, eta, tol = 1e-4, max_count = 10000):
    w = [w_init] 
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    while count < max_count:
 # mix data for stochastic gradient descent method
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
            # stopping criteria
            if count%check_w_after == 0: 
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

print("b")
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 350, shuffle = False)
Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

LinReg = LinearRegression(fit_intercept = False)
LinReg.fit(Xbar, y_train)
Xtest = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
y_pred = LinReg.predict(Xtest)
print("MSE = {}".format(mean_squared_error(y_test, y_pred)))

print("c")
model_sk = GaussianNB()
X_train, X_test, y_train, y_test = train_test_split(X, y_labels, train_size = 350, shuffle = False)

model_sk.fit(X_train, y_train)
y_pred = model_sk.predict(X_test)
print("Accuracy = ", accuracy_score(y_test, y_pred))
print("Precision = ", precision_score(y_test, y_pred))
print("Recall = ", recall_score(y_test, y_pred))