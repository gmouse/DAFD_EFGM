import pickle
import keras
def LoadData_pickle(path,name,type='rb'):
  with open(path+name+'.pkl', type) as f:
      data = pickle.load(f)
  return data

# def save_data(x,class_num,name):
#     if name=='train':
#         file = open('./data/preprocessed/C' + class_num + '_train_pre'  + '.pkl', 'wb')
#         print(x.shape, 'Save data to the path:', './data/preprocessed/C' + class_num+'_train_pre'  + '.pkl')
#     else:
#         file = open('./data/preprocessed/C' + class_num + '_test_pre'  + '.pkl', 'wb')
#         print(x.shape, 'Save data to the path:', './data/preprocessed/C' + class_num + '_test_pre' + '.pkl')
#     pickle.dump(x, file)
#     file.close()


def leaky_relu(features, alpha=0.2):
        f1 = 0.5 * (1 + alpha)
        f2 = 0.5 * (1 - alpha)
        return f1 * features + f2 * keras.backend.abs(features)


