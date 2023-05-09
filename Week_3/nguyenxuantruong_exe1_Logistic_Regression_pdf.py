from __future__ import division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(2)
X = np.array([[0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 1.75, 2.00, 2.25,
             2.50, 2.75, 3.00, 3.25, 3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50]])
y = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1])
# extended data by adding a column of 1s (x_0 = 1)
X = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)

X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]

plt.plot(X0, y0, 'ro', markersize=8)
plt.plot(X1, y1, 'bs', markersize=8)
plt.show()


def sigmoid(s):
    return 1/(1 + np.exp(-s))


def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    # method to calculate model logistic regression by Stochastic Gradient Descent method
    # eta: learning rate; tol: tolerance; max_count: maximum iterates
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
# loop of stochastic gradient descent
    while count < max_count:
     # shuffle the order of data (for stochastic gradient descent).
     # and put into mix_id
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta*(yi - zi)*xi
            count += 1
# stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

eta = .05
d = X.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(X, y, w_init, eta)
print(w[-1])

print(sigmoid(np.dot(w[-1].T, X)))

X0 = X[1, np.where(y == 0)][0]
y0 = y[np.where(y == 0)]
X1 = X[1, np.where(y == 1)][0]
y1 = y[np.where(y == 1)]
plt.plot(X0, y0, 'ro', markersize=8)
plt.plot(X1, y1, 'bs', markersize=8)
xx = np.linspace(0, 6, 1000)
w0 = w[-1][0][0]
w1 = w[-1][1][0]
threshold = -w0/w1
yy = sigmoid(w0 + w1*xx)
plt.axis([-2, 8, -1, 2])
plt.plot(xx, yy, 'g-', linewidth=2)
plt.plot(threshold, .5, 'y^', markersize=8)
plt.xlabel('studying hours')
plt.ylabel('predicted probability of pass')
plt.show()

Xb = np.array([[2.45, 1.85, 3.75, 3.21, 4.05]])
Xb = np.concatenate((np.ones((1, Xb.shape[1])), Xb), axis=0)
results = []
for i in sigmoid(np.dot(w[-1].T, Xb))[0]:
    if (i < 0.5):
        results.append("fail")
    if (i >= 0.5):
        results.append("success")
print(results)

# use library
logReg = LogisticRegression(penalty="none")
logReg.fit(X.T, y.T)
print(logReg.coef_)
y_expected = logReg.predict(Xb.T)
print(logReg.predict(Xb.T))
y_actual = np.array([1, 1, 0, 0, 0])
print(accuracy_score(y_actual, y_expected))