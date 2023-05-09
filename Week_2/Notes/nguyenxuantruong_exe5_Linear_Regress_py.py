import numpy as np
import math
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

path = "D:/Code/Python/MachineLearning/Week_2/"

with open(path + "real_estate.csv") as f:
    lines  = f.readlines()

x_data = []
y_data = []
lines.pop(0)
for l in lines:
    splitted = l.replace('\n', '').split(',')
    splitted.pop(0)
    splitted = list(map(float, splitted))
    y_data.append([splitted[-1]])
    x_data.append([int(splitted[0]), int(splitted[1]), splitted[2], splitted[3], splitted[4], splitted[5]])

x_data = np.asarray(x_data)
y_data = np.asarray(y_data)
one = np.ones((x_data.shape[0], 1))
Xbar = np.concatenate((one, x_data), axis=1)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, y_data)

train_label = np.array(y_data[:350,])
train_data = np.array(x_data[:350,])

test_label = np.array(np.asarray(y_data)[350:,])
test_data = np.array(np.asarray(x_data)[350:,])

one_train = np.ones((train_data.shape[0], 1))
Xbar_train = np.concatenate((one_train, train_data), axis=1)
regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar_train, train_label)

one_test = np.ones((test_data.shape[0], 1))
Xbar_test = np.concatenate((one_test, test_data), axis=1)
predict = regr.predict(Xbar_test)
print(mean_squared_error(predict, test_label) * predict.shape[0])