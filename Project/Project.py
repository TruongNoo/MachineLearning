import gzip
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
# 1
# Đọc dữ liệu
# Đọc dữ liệu ảnh train
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

# Đọc dữ liệu nhãn train
with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = np.frombuffer(f.read(), np.uint8, offset=8)

# Đọc dữ liệu ảnh validation
with gzip.open('t10k-images-idx3-ubyte.gz', 'rb') as f:
    val_images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 784)

# Đọc dữ liệu nhãn validation
with gzip.open('t10k-labels-idx1-ubyte.gz', 'rb') as f:
    val_labels = np.frombuffer(f.read(), np.uint8, offset=8)
    
pca = PCA(n_components=3) #Giảm số chiều xuống 3D
train_images_pca = pca.fit_transform(train_images)

# Hiển thị dữ liệu trên không gian 2D hoặc 3D
fig = plt.figure()
if pca.n_components == 2:
    plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels, cmap='viridis')
elif pca.n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(train_images_pca[:, 0], train_images_pca[:, 1], train_images_pca[:, 2], c=train_labels, cmap='viridis')
plt.show()

# 4
# Áp dụng PCA
val_images_pca = pca.transform(val_images)

# Huấn luyện mô hình Gaussian Naive Bayes trên dữ liệu train
nb = GaussianNB()
nb.fit(train_images, train_labels)

# Dự đoán nhãn trên dữ liệu validation
val_pred = nb.predict(val_images)

# Hiển thị các kết quả
print("Use Gaussian Naive Bayes model:")
print("Accuracy:", accuracy_score(val_labels, val_pred))
print("Confusion matrix:\n", confusion_matrix(val_labels, val_pred))
print("Precision:", precision_score(val_labels, val_pred, average='macro'))
print("Recall:", recall_score(val_labels, val_pred, average='macro'))

# Huấn luyện mô hình Multinomial Naive Bayes trên dữ liệu train
nb = MultinomialNB()
nb.fit(train_images, train_labels)

# Dự đoán nhãn trên dữ liệu validation
val_pred = nb.predict(val_images)

# Hiển thị các kết quả
print("----------------------------------")
print("Use Multinomial Naive Bayes model:")
print("Accuracy:", accuracy_score(val_labels, val_pred))
print("Confusion matrix:\n", confusion_matrix(val_labels, val_pred))
print("Precision:", precision_score(val_labels, val_pred, average='macro'))
print("Recall:", recall_score(val_labels, val_pred, average='macro'))

# Huấn luyện mô hình Bernoulli Naive Bayes trên dữ liệu train
nb = BernoulliNB()
nb.fit(train_images, train_labels)

# Dự đoán nhãn trên dữ liệu validation
val_pred = nb.predict(val_images)

# Hiển thị các kết quả
print("----------------------------------")
print("Use Bernoulli Naive Bayes model:")
print("Accuracy:", accuracy_score(val_labels, val_pred))
print("Confusion matrix:\n", confusion_matrix(val_labels, val_pred))
print("Precision:", precision_score(val_labels, val_pred, average='macro'))
print("Recall:", recall_score(val_labels, val_pred, average='macro'))