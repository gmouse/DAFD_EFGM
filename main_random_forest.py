import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
import csv

def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
          data = pickle.load(f)
  return data

x0, _ = LoadData_pickle(path='./data/data_feature/', name='C0_train', type='rb')
x0_t, _ = LoadData_pickle(path='./data/data_feature/', name='C0_test', type='rb')
x1, _ = LoadData_pickle(path='./data/data_feature/', name='C1_train', type='rb')
x1_t, _ = LoadData_pickle(path='./data/data_feature/', name='C1_test', type='rb')
x2, _ = LoadData_pickle(path='./data/data_feature/', name='C2_train', type='rb')
x2_t, _ = LoadData_pickle(path='./data/data_feature/', name='C2_test', type='rb')
x3, _ = LoadData_pickle(path='./data/data_feature/', name='C3_train', type='rb')
x3_t, _ = LoadData_pickle(path='./data/data_feature/', name='C3_test', type='rb')

data_x = np.vstack((x0, x1, x2, x3))
num = 700
data_y = np.array([0] * num + [1] * num + [2] * num + [3] * num).reshape(num * 4, 1)

data_x_T = np.vstack((x0_t, x1_t, x2_t, x3_t))
num_t = 300
data_y_T = np.array([0] * num_t + [1] * num_t + [2] * num_t + [3] * num_t).reshape(num_t * 4, 1)

clf = RandomForestClassifier(500)
# clf = svm.SVC(kernel='rbf')
clf.fit(data_x, data_y)
labels_pred = clf.predict(data_x_T)
acc = accuracy_score(data_y_T, labels_pred)
print('Original data')
print(confusion_matrix(data_y_T, labels_pred))
print(acc)

# Generated data
path = './data/data_generated'
g1 = LoadData_pickle(path=path, name='C1_Generated', type='rb')[0:700, :]
g2 = LoadData_pickle(path=path, name='C2_Generated', type='rb')[0:700, :]
g3 = LoadData_pickle(path=path, name='C3_Generated', type='rb')[0:700, :]
# print(g1.mean(),g1.std())
# print(g2.mean(),g2.std())
 # print(g3.mean(),g3.std())
gdata_x = np.vstack((x0, g1, g2, g3))
num_g = 700
gdata_y = np.array([0] * num_g + [1] * num_g + [2] * num_g + [3] * num_g).reshape(num_g * 4, 1)

clf = RandomForestClassifier(500)
# clf = svm.SVC(kernel='rbf')
clf.fit(gdata_x, gdata_y)
labels_pred_g = clf.predict(data_x_T)
acc_g = accuracy_score(data_y_T, labels_pred_g)
print('Generated data')
print(confusion_matrix(data_y_T, labels_pred_g))
print(acc_g)

