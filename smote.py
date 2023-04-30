import tensorflow as tf
import numpy as np
from dataset_provider import provide_data
from np_to_tfrecords import np_to_tfrecords

faults = [0,1,2,3]
path = '../dataset/'
train_set = {}
features_g = []
labels_g = []

for i in faults:
    if i<1:
        train_set[i] = provide_data(path, 14000, 'wpt/balance/C' + str(i) + '_b')
    else:
        train_set[i] = provide_data(path, 140, 'wpt/imbalance/C' + str(i))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in faults:
        featuresg_, labelsg_ = sess.run(train_set[i])
        features_g.extend(featuresg_)
        labels_g.extend(labelsg_)

features = np.array(features_g)
labels = np.array(labels_g,'int64')
print(features.shape,labels.shape)

from collections import Counter

y=labels.reshape(730,)
counter = Counter(y)
print(counter)

from imblearn.over_sampling import SMOTE

oversample=SMOTE(sampling_strategy='auto',random_state=None,k_neighbors=7)
train_x,train_y=oversample.fit_resample(features,y)
counter = Counter(train_y)
print(counter)
train_y = train_y.reshape(2800, 1)
np_to_tfrecords(train_x, train_y, '../dataset/wpt/fitting/train_smote', verbose=True)


#
# print(features.shape,labels.shape)
# print(train_x.shape,train_y.shape)
