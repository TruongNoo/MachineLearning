import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model

data = pd.read_csv("D:/Code/Python/MachineLearning/Week_2/SAT_GPA.csv")
data.describe()

y = data['GPA']
x = data['SAT']

plt.scatter(x,y)
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()

train_label = np.array([np.asarray(y)[:60,]]).T
validation_label = np.array([np.asarray(y)[60:,]]).T
train_data = np.array([np.asarray(x)[:60,]]).T
validation_data = np.array([np.asarray(x)[60:,]]).T

one = np.ones((train_data.shape[0],1))
Xbar = np.concatenate((one, train_data), axis=1)

regr = linear_model.LinearRegression(fit_intercept=False)
regr.fit(Xbar, train_label)
w = regr.coef_
print(w)

one = np.ones((validation_data.shape[0],1))
Xbar_test = np.concatenate((one, validation_data), axis=1)
predict = regr.predict(Xbar_test)
print(mean_squared_error(validation_label, predict))

t_0 = w[0][0]
t_1 = w[0][1]
plt.scatter(x,y)
y1 = t_1*x + t_0
fig = plt.plot(x,y1, lw=4, c='orange', label = 'regression line')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('GPA', fontsize = 20)
plt.show()