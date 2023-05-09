import pandas as pd
import numpy as np
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import BernoulliNB

# change to your data's path
data = pd.read_csv('banking.csv')
data.head()

# convert field of 'month'
dict_month = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
              'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
data['month'] = data['month'].map(dict_month)

# convert field of dayOfweek
dict_day = {'sun': 1, 'mon': 2, 'tue': 3,
            'wed': 4, 'thu': 5, 'fri': 6, 'sat': 7}
data['day_of_week'] = data['day_of_week'].map(dict_day)

# conver binary fields
# default :
data.default.replace({'no': 0, 'yes': 1, 'unknown': 2}, inplace=True)
# housing :
data.housing.replace({'no': 0, 'yes': 1, 'unknown': 2}, inplace=True)
# loan :
data.loan.replace({'no': 0, 'yes': 1, 'unknown': 2}, inplace=True)


# convert categories field by one host coding
marital_dummies = pd.get_dummies(data['marital'], prefix='marital')
marital_dummies.drop('marital_divorced', axis=1, inplace=True)
data = pd.concat([data, marital_dummies], axis=1)

job_dummies = pd.get_dummies(data['job'], prefix='job')
job_dummies.drop('job_unknown', axis=1, inplace=True)
data = pd.concat([data, job_dummies], axis=1)

education_dummies = pd.get_dummies(data['education'], prefix='education')
education_dummies.drop('education_unknown', axis=1, inplace=True)

data = pd.concat([data, education_dummies], axis=1)
contact_dummies = pd.get_dummies(data['contact'], prefix='contact')
#contact_dummies.drop('contact_unknown', axis=1, inplace=True)
data = pd.concat([data, contact_dummies], axis=1)
poutcome_dummies = pd.get_dummies(data['poutcome'], prefix='poutcome')
#poutcome_dummies.drop('poutcome_unknown', axis=1, inplace=True)
data = pd.concat([data, poutcome_dummies], axis=1)

data['pdays'] = data['pdays'].apply(lambda row: 0 if row == -1 else 1)

data.drop(['job', 'education', 'marital', 'contact',
          'poutcome'], axis=1, inplace=True)


X = data.drop(['y'], axis=1)
y = data['y']

# Chia dữ liệu thành phần Training và Test theo tỷ lệ 8:2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False)

# a: Su dung mo hinh hoi quy logistic
Xbar = np.concatenate((np.ones((X_train.shape[0], 1)), X_train), axis=1)
start_time = time.time()

logReg = linear_model.LogisticRegression()

logReg.fit(Xbar, y_train)

Xbar2 = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis=1)
y_pred = logReg.predict(Xbar2)
logReg.fit(Xbar2, y_pred)
print(logReg.coef_)
end_time = time.time()
print("---------------------------------------------")

print('Accuracy_score: ', accuracy_score(y_pred, y_test))
print('Recall_score: ', recall_score(y_pred, y_test))
print('Precision_score: ', precision_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))

elapsed_time = end_time - start_time
print("Elapsed_time:{0}".format(elapsed_time) + "[sec]")
print("---------------------------------------------")

# b. su dung mo hing navie bayes
start_time = time.time()

clf = BernoulliNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
clf.fit(X_test, y_pred)

end_time = time.time()

print('Accuracy_score: ', accuracy_score(y_pred, y_test))
print('Recall_score: ', recall_score(y_pred, y_test))
print('Precision_score: ', precision_score(y_pred, y_test))
print('F1-score: ', f1_score(y_pred, y_test))

elapsed_time = end_time - start_time
print("Elapsed_time:{0}".format(elapsed_time) + "[sec]")