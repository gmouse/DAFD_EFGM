# import tensorflow as tf
import numpy as np
from dataset_provider import provide_data
from np_to_tfrecords import np_to_tfrecords
from sklearn import preprocessing
from tfrecords_builder import signal2wp_energy
import h5py
import pywt

np.random.seed(130)
# 读取数据
import scipy.io as sio
myvar = sio.loadmat('./dataset/C00_C01_C03_C08_C10_C15_C30_S50_L1_Ch_01_to_06.mat')
myvar = myvar['Y'][:,3072:6144]

# 标准化standscaler()
from sklearn import preprocessing
stand_means = preprocessing.StandardScaler()
myvar = stand_means.fit_transform(myvar)
print(myvar.max(),myvar.min())
print(myvar.mean(),myvar.std())

# # 小波包变换，db7，分解7层
wpt=[]
for k in range(myvar.shape[0]):
    wpt.append(signal2wp_energy(myvar[k,:],wavelets=['db7'],max_level=7))

wpt=np.array(wpt).reshape(myvar.shape[0],-1)
# wpt=np.array(wpt).reshape(myvar.shape[0],128)

#划分类别
C00=wpt[0:1000,:]
C03=wpt[2000:3000,:]
C15=wpt[5000:6000,:]
C30=wpt[6000:7000,:]
del myvar

#标签
y=[]
for i in range(4):
    y.append(np.array([i]*1000))
y=np.array(y).reshape(4000,1)

x00 = C00
x03 = C03
x15 = C15
x30 = C30

# Shuffle
mix0 = [i for i in range(len(x00))]
np.random.shuffle(mix0)
x_0 = x00[mix0]

mix3 = [i for i in range(len(x03))]
np.random.shuffle(mix3)
x_3 = x03[mix3]

mix15 = [i for i in range(len(x15))]
np.random.shuffle(mix15)
x_15 = x15[mix15]

mix30 = [i for i in range(len(x30))]
np.random.shuffle(mix30)
x_30 = x30[mix30]

del C00,C03,C15,C30
del x00,x03,x15,x30

# 划分测试集与数据集
from sklearn.model_selection import train_test_split
x0_train,x0_test,y0_train,y0_test=train_test_split(x_0,y[0:1000,:],test_size=0.3)
x3_train,x3_test,y3_train,y3_test=train_test_split(x_3,y[1000:2000,:],test_size=0.3)
x15_train,x15_test,y15_train,y15_test=train_test_split(x_15,y[2000:3000],test_size=0.3)
x30_train,x30_test,y30_train,y30_test=train_test_split(x_30,y[3000:4000],test_size=0.3)
del x_0,x_3,x_15,x_30
del mix0,mix3,mix15,mix30

train_x=np.vstack([x0_train,x3_train[0:10,:],x15_train[0:10,:],x30_train[0:10,:]])
train_y=np.vstack([y0_train,y3_train[0:10,:],y15_train[0:10,:],y30_train[0:10,:]])
test_x=np.vstack([x0_test,x3_test,x15_test,x30_test])
test_y=np.vstack([y0_test,y3_test,y15_test,y30_test])

print(train_x.shape,train_y.shape)
print(test_x.shape,test_y.shape)

mix = [i for i in range(len(train_x))]
np.random.shuffle(mix)
train_x = train_x[mix]
train_y = train_y[mix]

# # 3,Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
cart_model = RandomForestClassifier(100)
cart_model.fit(train_x, train_y)
result = cart_model.score(test_x, test_y)
labels_pred = cart_model.predict(test_x)
acc=accuracy_score(test_y,labels_pred)
print('The acc is ',acc)
print(classification_report(test_y,labels_pred,output_dict=True))
print(confusion_matrix(test_y,labels_pred))

np_to_tfrecords(x0_train[0:10],  y0_train[0:10], '../dataset/imbalance/C0', verbose=True)
np_to_tfrecords(x3_train[0:10],  y3_train[0:10], '../dataset/imbalance/C1', verbose=True)
np_to_tfrecords(x15_train[0:10], y15_train[0:10], '../dataset/imbalance/C2', verbose=True)
np_to_tfrecords(x30_train[0:10], y30_train[0:10], '../dataset/imbalance/C3', verbose=True)


np_to_tfrecords(x0_test, y0_test, '../dataset/balance/C0_test', verbose=True)
np_to_tfrecords(x3_test, y3_test, '../dataset/balance/C3_test', verbose=True)
np_to_tfrecords(x15_test, y15_test, '../dataset/balance/C15_test', verbose=True)
np_to_tfrecords(x30_test, y30_test, '../dataset/balance/C30_test', verbose=True)

np_to_tfrecords(x0_train, y0_train,   '../dataset/balance/C0_b', verbose=True)
np_to_tfrecords(x3_train, y3_train,   '../dataset/balance/C3_b', verbose=True)
np_to_tfrecords(x15_train, y15_train, '../dataset/balance/C15_b', verbose=True)
np_to_tfrecords(x30_train, y30_train, '../dataset/balance/C30_b', verbose=True)

import pickle
with open('../dataset/imbalance/C0.pkl','wb') as f:
    pickle.dump([x0_train[0:10],y0_train[0:10]],f,pickle.HIGHEST_PROTOCOL)
with open('../dataset/imbalance/C1.pkl','wb') as f:
    pickle.dump([x3_train[0:10],y3_train[0:10]],f,pickle.HIGHEST_PROTOCOL)
with open('../dataset/imbalance/C2.pkl','wb') as f:
    pickle.dump([x15_train[0:10],y15_train[0:10]],f,pickle.HIGHEST_PROTOCOL)
with open('../dataset/imbalance/C3.pkl','wb') as f:
    pickle.dump([x30_train[0:10],y30_train[0:10]],f,pickle.HIGHEST_PROTOCOL)
