import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import Callback
from datetime import datetime
from flow_layers import *
from util import LoadData_pickle
from NN_architecture import build_basic_model

# load dataset
class_num = '3'
data_T,_ = LoadData_pickle(path='./data/data_feature/',name='C'+class_num,type='rb')
data_t,_ = LoadData_pickle(path='./data/data_feature/',name='C'+class_num+'_test',type='rb')
data_T = data_T.reshape(10,128)
data_t = data_t.reshape(300,128)

original_dim = 128
depth = 8  # orginal paper use depth=32
level = 3  # orginal paper use level=6 for 256*256 CelebA HQ
batch_size = 32
epochs = 1000

x_in = Input(shape=(original_dim,))  # add noise into inputs for stability.
x_noise = Lambda(lambda s: K.in_train_phase(s + 0.01 * K.random_uniform(K.shape(s)), s))(x_in)
x = Reshape((-1,1, original_dim, 1))(x_noise)
x_outs = []

squeeze = Squeeze()
inner_layers = []
outer_layers = []
for i in range(5):
    inner_layers.append([])
for i in range(3):
    outer_layers.append([])

for i in range(level):
    x = squeeze(x)
    for j in range(depth):
        actnorm = Actnorm()
        permute = Permute(mode='random')
        split = Split()
        couple = CoupleWrapper(build_basic_model(2**(i+1)))
        concat = Concat()
        inner_layers[0].append(actnorm)
        inner_layers[1].append(permute)
        inner_layers[2].append(split)
        inner_layers[3].append(couple)
        inner_layers[4].append(concat)
        x = actnorm(x)
        x = permute(x)
        x1, x2 = split(x)
        x1, x2 = couple([x1, x2])
        x = concat([x1, x2])
    if i < level-1:
        split = Split()
        condactnorm = CondActnorm()
        reshape = Reshape()
        outer_layers[0].append(split)
        outer_layers[1].append(condactnorm)
        outer_layers[2].append(reshape)
        x1, x2 = split(x)
        x_out = condactnorm([x2, x1])
        x_out = reshape(x_out)
        x_outs.append(x_out)
        x = x1
    else:
        for _ in outer_layers:
            _.append(None)

final_actnorm = Actnorm()
final_concat = Concat()
final_reshape = Reshape()

x = final_actnorm(x)
x = final_reshape(x)
x = final_concat(x_outs+[x])

encoder = Model(x_in, x)
for l in encoder.layers:
    if hasattr(l, 'logdet'):
        encoder.add_loss(l.logdet)

encoder.summary()
encoder.compile(loss=lambda y_true,y_pred: 0.5 * K.sum(y_pred**2, 1) + 0.5 * np.log(2*np.pi) * K.int_shape(y_pred)[1],
                optimizer=Adam(1e-4))
class Evaluate(Callback):
    def __init__(self):
        super().__init__()
        self.lowest = 1e10
    def on_epoch_end(self, epoch, logs=None):
        if epoch == epochs - 1 :
            count_examples = 0
            num_examples_to_eval = 700
            gen_list = []
            while count_examples < num_examples_to_eval:
                x_decoded = decoder.predict(np.random.randn(batch_size, 128), verbose=1)
                gen_list.extend(x_decoded)
                count_examples += batch_size
            generated_examples = np.array(gen_list)
            gentime = datetime.strftime(datetime.now(), '%m%d %H%M%S')
            print(generated_examples.shape)
            file = open('./data/data_generated/C'+class_num+'_Generated_'+gentime+'.pkl', 'wb')
            pickle.dump(generated_examples, file)
            file.close()

x_in = Input(shape=K.int_shape(encoder.outputs[0])[1:])
x = x_in

x = final_concat.inverse()(x)
outputs = x[:-1]
x = x[-1]
x = final_reshape.inverse()(x)
x = final_actnorm.inverse()(x)
x1 = x

for i,(split,condactnorm,reshape) in enumerate(list(zip(*outer_layers))[::-1]):
    if i > 0:
        x1 = x
        x_out = outputs[-i]
        x_out = reshape.inverse()(x_out)
        x2 = condactnorm.inverse()([x_out, x1])
        x = split.inverse()([x1, x2])
    for j,(actnorm,permute,split,couple,concat) in enumerate(list(zip(*inner_layers))[::-1][i*depth: (i+1)*depth]):
        x1, x2 = concat.inverse()(x)
        x1, x2 = couple.inverse()([x1, x2])
        x = split.inverse()([x1, x2])
        x = permute.inverse()(x)
        x = actnorm.inverse()(x)
    x = squeeze.inverse()(x)
x = Reshape(shape=(-1,original_dim))(x)

decoder = Model(x_in, x)
decoder.summary()
evaluator = Evaluate()
history = encoder.fit(data_T,
      data_T,
      batch_size = batch_size,
      epochs = epochs,
      validation_data=(data_t, data_t),
      callbacks=[evaluator],
      )

# summarize history for loss
gentime_1 = datetime.strftime(datetime.now(), '%m%d %H%M%S')
plt.figure(1)
plt.plot()
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig('./data/loss_img/C'+class_num+'_Generated_'+gentime_1+'.png')
plt.show()
