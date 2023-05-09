import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#Read csv data file, change to your location
df = pd.read_csv("D:\Code\Python\MachineLearning\Week_5\iris.csv")

df
df.describe()
sns.pairplot(df, hue = "variety")

# We're seperating the species column
species = df["variety"].tolist()
X = df.drop("variety", 1)
# Standardize the data
X = (X - X.mean()) / X.std(ddof=0)
# Calculating the correlation matrix of the data
X_corr = (1 / 150) * X.T.dot(X)
# Plotting the correlation matrix
plt.figure(figsize=(10,10))
sns.heatmap(X_corr, vmax=1, square=True,annot=True)
plt.title('Correlation matrix')

# method1
u,s,v = np.linalg.svd(X_corr)
eig_values, eig_vectors = s, u
eig_values, eig_vectors
# method2
np.linalg.eig(X_corr)

# plotting the variance explained by each PC 
explained_variance=(eig_values / np.sum(eig_values))*100
plt.figure(figsize=(8,4))
plt.bar(range(4), explained_variance, alpha=0.6)
plt.ylabel('Percentage of explained variance')
plt.xlabel('Dimensions')

pc1 = X.dot(eig_vectors[:,0])
pc2 = X.dot(eig_vectors[:,1])

# plotting in 2D
def plot_scatter(pc1, pc2):
    fig, ax = plt.subplots(figsize=(15, 8))
 
    species_unique = list(set(species))
    species_colors = ["r","b","g"]
 
    for i, spec in enumerate(species):
        plt.scatter(pc1[i], pc2[i], label = spec, s = 20, c=species_colors[species_unique.index(spec)])
        ax.annotate(str(i+1), (pc1[i],pc2[i]))
 
    from collections import OrderedDict
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), prop={'size': 15}, loc=4)
 
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.axhline(y=0, color="grey", linestyle="--")
    ax.axvline(x=0, color="grey", linestyle="--")
 
    plt.grid()
    plt.axis([-4, 4, -3, 3])
    plt.show()
 
plot_scatter(pc1, pc2)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = np.array([pc1,pc2]).T
y = species = df["variety"].tolist()

from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit(df.variety)
df["variety"]=le.transform(df.variety)

y = np.array(df["variety"])

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, shuffle = False)
Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)

LogReg = LogisticRegression(max_iter = 1000, fit_intercept = False)
LogReg.fit(Xbar, y_train)
Xtest = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
y_pred = LogReg.predict(Xtest)
print("Confusion matrix: \n", str(LogReg.coef_))
print("Accuracy = ", accuracy_score(y_test, y_pred))