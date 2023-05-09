import math
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

path = ("D:/Code/Python/MachineLearning/Week_2/")
with open(path + "vidu4_lin_reg.txt") as f:
    lines  = f.readlines()
lines = [line.strip() for line in lines][1:]
lines = np.asarray([[float(a) for a in line.split(' ')] for line in lines])

x = lines[:, 1:-1]
y = lines[:, -1]
# Huấn luyện mô hình
one = np.ones((x.shape[0], 1))
Xbar = np.concatenate((one, x), axis = 1)
A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y)

w = np.dot(np.linalg.pinv(A), b)

print('w =', w)

# Chia dữ liệu thành tập train và tập test
x_train = x[:80,]
x_test = x[80:,]

y_train = y[:80]
y_test = y[80:]

# Huấn luyện mô hình
one = np.ones((x_train.shape[0], 1))
Xbar = np.concatenate((one, x_train), axis = 1)

A = np.dot(Xbar.T, Xbar)
b = np.dot(Xbar.T, y_train)

w = np.dot(np.linalg.pinv(A), b)

print('w =', w)

y_pred = [w[0] + np.dot(w[1:], a) for a in x_test]

error = np.abs(y_pred - y_test) # Sai số

print('Kỳ vọng của sai số: %f' % np.mean(error))
print('Phương sai của sai số: %f' % np.var(error))