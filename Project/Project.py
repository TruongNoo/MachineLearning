import gzip
import numpy as np
import random
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
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
    
fig, axs = plt.subplots(2, 5, figsize=(12, 6))
fig.tight_layout()

for label in range(10):
    # Lấy chỉ mục của ảnh có nhãn tương ứng
    indices = np.where(val_labels == label)[0]
    random_index = random.choice(indices)
    
    # Lấy ảnh và hiển thị
    image = val_images[random_index].reshape(28, 28)
    
    axs[label // 5, label % 5].imshow(image, cmap='gray')
    axs[label // 5, label % 5].set_title(f"Label: {label}")
    axs[label // 5, label % 5].axis('off')

plt.show()
    
pca = PCA(n_components=3)
train_images_pca = pca.fit_transform(train_images)

# Tạo một bảng màu với 10 màu tương ứng với 10 nhãn
cmap = ListedColormap(['green', 'orange', 'blue', 'red', 'purple', 'cyan', 'brown', 'pink', 'gray', 'yellow'])

# Hiển thị dữ liệu trên không gian 2D hoặc 3D
fig = plt.figure()
if pca.n_components == 2:
    scatter = plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels, cmap=cmap)
elif pca.n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(train_images_pca[:, 0], train_images_pca[:, 1], train_images_pca[:, 2], c=train_labels, cmap=cmap)

# Tạo colorbar
cbar = plt.colorbar(scatter, ticks=np.arange(10))
cbar.set_label('Labels')

plt.show()

# Giới hạn số lượng ảnh hiển thị là 5000
train_images_plt = train_images[:5000]
train_labels_plt = train_labels[:5000]

# Tạo một bảng màu với 10 màu tương ứng với 10 nhãn
cmap = ListedColormap(['green', 'orange', 'blue', 'red', 'purple', 'cyan', 'brown', 'pink', 'gray', 'yellow'])

# Áp dụng PCA
pca = PCA(n_components=3)
train_images_pca = pca.fit_transform(train_images_plt)

# Hiển thị dữ liệu trên không gian 2D hoặc 3D
fig = plt.figure()
if pca.n_components == 2:
    scatter = plt.scatter(train_images_pca[:, 0], train_images_pca[:, 1], c=train_labels_plt, cmap=cmap)
elif pca.n_components == 3:
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(train_images_pca[:, 0], train_images_pca[:, 1], train_images_pca[:, 2], c=train_labels_plt, cmap=cmap)

# Tạo colorbar
cbar = plt.colorbar(scatter, ticks=np.arange(10))
cbar.set_label('Labels')

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
print("Precision:", precision_score(val_labels, val_pred, average='micro'))
print("Recall:", recall_score(val_labels, val_pred, average='micro'))

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
print("Precision:", precision_score(val_labels, val_pred, average='micro'))
print("Recall:", recall_score(val_labels, val_pred, average='micro'))

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
print("Precision:", precision_score(val_labels, val_pred, average='micro'))
print("Recall:", recall_score(val_labels, val_pred, average='micro'))