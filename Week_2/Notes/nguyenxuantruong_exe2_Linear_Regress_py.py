import math
import numpy as np
from sklearn import datasets, linear_model

print("Trường hợp sử dụng Numpy")
path = 'D:/Code/Python/MachineLearning/Week_2/'
with open('fuel.txt') as f:
    lines = f.readlines()

x_data = []
y_data = []
lines.pop(0)
for line in lines:
    splitted = line.replace('\n', '').split(',')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    fuel = 1000 * splitted[1] / splitted[5]
    dlic = 1000 * splitted[0] / splitted[5]
    logMiles = math.log2(splitted[3])
    y_data.append([fuel])
    x_data.append([splitted[-1], dlic, splitted[2], logMiles])
x_data = np.asarray(x_data)
y_data = np.asarray(y_data)

def qr_householder(A):
    #""" Compute QR decomposition of A using Householder reflection"""
    M = A.shape[0]
    N = A.shape[1]
    # set Q to the identity matrix
    Q = np.identity(M)
    # set R to zero matrix
    R = np.copy(A)
    for n in range(N):
        # vector to transform
        x = A[n:, n]
        k = x.shape[0]
        # compute ro=-sign(x0)||x||
        ro = -np.sign(x[0]) * np.linalg.norm(x)
        
        # compute the householder vector v
        e = np.zeros(k)
        e[0] = 1
        v = (1 / (x[0] - ro)) * (x - (ro * e))
        # apply v to each column of A to find R
        for i in range(N):
            R[n:, i] = R[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ R[n:, i])
 
        # apply v to each column of Q
        for i in range(M):
           Q[n:, i] = Q[n:, i] - (2 / (v@v)) * ((np.outer(v, v)) @ Q[n:, i])
    return Q.transpose(), R
def linear_regression(x_data, y_data):
# """
# This function calculate linear regression base on x_data and y_data
# :param x_data: vector
# :param y_data: vector
# :return: w (regression estimate)
# """
 # add column 1
    x_bars = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis=1)
    Q, R = qr_householder(x_bars) # QR decomposition
    R_pinv = np.linalg.pinv(R) # calculate inverse matrix of R
    A = np.dot(R_pinv, Q.T) # apply formula
    return np.dot(A, y_data)

w = linear_regression(x_data, y_data) # get result
w = w.T.tolist()
line = ['Intercept', 'Tax', "Dlic", "Income", 'LogMiles']
res = list(zip(line, w[0]))
for o in res:
    print("{: >20}: {: >10}".format(*o))
print("Trường hợp sử dụng thư viện Scikit-Learn")
Xbar = np.c_[x_data, np.ones(len(x_data))]
regr = linear_model.LinearRegression(fit_intercept=False) 
# fit_intercept = False for calculating the bias
regr.fit(Xbar, y_data)
print(regr.coef_)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_data)
w = np.dot(np.linalg.pinv(A), b)
print(w)