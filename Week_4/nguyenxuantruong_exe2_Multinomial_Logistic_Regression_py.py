from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.decomposition import PCA as sklearnPCA
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from scipy import sparse
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import random
from sklearn.model_selection import train_test_split
# import some data to play with
C = 3
iris = datasets.load_iris()
X = iris.data[:, :4]  # we take full 4 features
Y = iris.target
# Normalize data
X_norm = (X - X.min())/(X.max() - X.min())
pca = sklearnPCA(n_components=2)  # 2-dimensional PCA
transformed = pd.DataFrame(pca.fit_transform(X_norm))
plt.axis("off")
plt.scatter(transformed[Y == 0][0], transformed[Y == 0]
            [1], s=9, label='IRIS Setosa', c='red')
plt.scatter(transformed[Y == 1][0], transformed[Y == 1][1],
            s=9, label='IRIS Versicolor', c='green', marker="^")

plt.scatter(transformed[Y == 2][0], transformed[Y == 2]
            [1], s=9, label='IRIS Virginica', c='blue', marker="s")
plt.legend()
plt.show()

iris = load_iris()
# print(iris)
X = iris.data  # Observed variable
Y = iris.target  # Dependent variable (label)

# print(X.shape)
# print(Y.shape)
# Splitting Train and test Data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
# sc=StandardScaler()
# X_train=sc.fit_transform(X_train)
# X_test=sc.transform(X_test)
# Call to Logistic Regression Model - SAG: solving is based on Stochastic Average Gradient
lorg = LogisticRegression(multi_class='multinomial',
                          solver='sag', max_iter=5000)
# and train model by Training Dataset
lorg.fit(X_train, Y_train)
# Then Predict the Test data
Y_pred = lorg.predict(X_test)
# for accuracy
print("Accuracy=", accuracy_score(Y_test, Y_pred))
# for confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)


def convert_labels(y, C=C):
    Y = sparse.coo_matrix(
        (np.ones_like(y), (y, np.arange(len(y)))), shape=(C, len(y))).toarray()
    return Y
# Y = convert_labels(y, C)


def softmax_stable(Z):
    e_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    A = e_Z / e_Z.sum(axis=0)
    return A


def softmax(Z):
    e_Z = np.exp(Z)
    A = e_Z / e_Z.sum(axis=0)
    return A


def softmax_regression(X, y, W_init, eta, tol=1e-4, max_count=10000):
    W = [W_init]
    C = W_init.shape[1]
    Y = convert_labels(y, C)
    it = 0
    N = X.shape[1]
    d = X.shape[0]

    count = 0
    check_w_after = 20
    while count < max_count:
        # mix data
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = Y[:, i].reshape(C, 1)
            ai = softmax(np.dot(W[-1].T, xi))
            W_new = W[-1] + eta*xi.dot((yi - ai).T)
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(W_new - W[-check_w_after]) < tol:
                    return W
            W.append(W_new)
    return W
# cost or loss function


def cost(X, Y, W):
    A = softmax(W.T.dot(X))
    return -np.sum(Y*np.log(A))
# Predict that X belong to which class (1..C now indexed as 0..C-1 )


def pred(W, X):
    A = softmax_stable(W.T.dot(X))
    return np.argmax(A, axis=0)


Xbar = np.array(np.concatenate(
    (np.ones((X_train.shape[0], 1)), X_train), axis=1)).T

eta = .05
d = Xbar.shape[0]
W_init = np.random.randn(Xbar.shape[0], C)
W = softmax_regression(Xbar, Y_train, W_init, eta)
print(W[-1])

Xtest = np.array(np.concatenate(
    (np.ones((X_test.shape[0], 1)), X_test), axis=1)).T
y_pred = pred(W[-1], Xtest)
print("Accuracy=", accuracy_score(Y_test, Y_pred))
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
