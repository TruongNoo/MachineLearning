import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

path = 'D:/Code/Python/MachineLearning/Week_1/data1/'

def create_data():
  with open(path + 'datacum.txt') as f:
    _data = f.readlines()
  test_file = open(path + 'test.txt', 'w')
  train_file = open(path + 'train.txt', 'w')
  num_benign = 0
  num_malignant = 0
  for i, line in enumerate(_data):
    a = line.strip().split('\n')
    
    if '#####' in a[0] or a[0] == '':
      continue
    if ',0,' in a[0]:
      continue

    point = a[0].split(',')
    if (num_benign >= 80 and int(point[1]) == 2) or (num_malignant >= 40 and int(point[1]) == 4):
      train_file.write(a[0] + '\n')
    
    if (num_benign < 80 and int(point[1]) == 2) or (num_malignant < 40 and int(point[1]) == 4):
      if int(point[1]) == 2:
          num_benign += 1
      else:
          num_malignant += 1
      test_file.write(a[0] + '\n')

def read_data(file_name):
  with open(path + file_name) as f:
    _input_data = f.readlines()
  
  _labels = [int(point.strip().split(',')[1]) for point in _input_data]

  raw_data = [point.strip() for point in _input_data]
  _data = np.zeros((len(raw_data), 9), dtype=int)
  for i, line in enumerate(raw_data):
    a = line.split(',')
    a = [int(b) for b in a]
    _data[i, :] = np.array(a[2:])
  return _data, _labels
create_data()
train_data_file = 'train.txt'
test_data_file = 'test.txt'

train_data, train_labels = read_data(train_data_file)
test_data, test_labels = read_data(test_data_file)

model = GaussianNB()
model.fit(train_data, train_labels)

#Run 
result_predict = model.predict(test_data)

print('Accuracy: %.2f%%' % (accuracy_score(test_labels, result_predict) * 100))
print('Recall: %.2f%%' % (recall_score(test_labels, result_predict, pos_label=2) * 100))
print('Precision: %.2f%%' % (precision_score(test_labels, result_predict, pos_label=2) * 100))
print(confusion_matrix(test_labels, result_predict))