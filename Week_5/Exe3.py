import numpy as np
import gzip
from time import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Đọc dữ liệu ảnh từ file
with gzip.open('train-images-idx3-ubyte.gz', 'rb') as f:
    images = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)

# Đọc nhãn từ file
with gzip.open('train-labels-idx1-ubyte.gz', 'rb') as f:
    labels = np.frombuffer(f.read(), np.uint8, offset=8)

# Lấy một tập dữ liệu con gồm 5000 ảnh bất kỳ
np.random.seed(42)
idx = np.random.choice(images.shape[0], size=5000, replace=False)
images = images[idx]
labels = labels[idx]

# Tạo các bộ dữ liệu đã qua giảm chiều và chưa qua giảm chiều với tỉ lệ train:validation là 0.7:0.3
pca = PCA(n_components=100)
images_reduced = pca.fit_transform(images)
X_train, X_val, y_train, y_val = train_test_split(images_reduced, labels, test_size=0.3, random_state=42)
X_train_orig, X_val_orig, y_train_orig, y_val_orig = train_test_split(images, labels, test_size=0.3, random_state=42)

# Huấn luyện và đánh giá độ chính xác trên dữ liệu đã qua giảm chiều
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs')
start_time = time()
clf.fit(X_train, y_train)
train_time_pca = time() - start_time
start_time = time()
y_pred = clf.predict(X_val)
test_time_pca = time() - start_time
accuracy_pca = accuracy_score(y_val, y_pred)

# Huấn luyện và đánh giá độ chính xác trên dữ liệu nguyên bản
clf_orig = LogisticRegression(multi_class='multinomial', solver='lbfgs')
start_time = time()
clf_orig.fit(X_train_orig, y_train_orig)
train_time_orig = time() - start_time
start_time = time()
y_pred_orig = clf_orig.predict(X_val_orig)
test_time_orig = time() - start_time
accuracy_orig = accuracy_score(y_val_orig, y_pred_orig)

print(f'Accuracy on PCA-reduced data: {accuracy_pca:.2f} (train time: {train_time_pca:.2f}s, test time: {test_time_pca:.2f}s)')
print(f'Accuracy on original data: {accuracy_orig:.2f} (train time: {train_time_orig:.2f}s, test time: {test_time_orig:.2f}s)')