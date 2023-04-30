import numpy as np
from multiprocessing import Pool
import re
import scipy.io as sio
import os
from os.path import join
import scipy.stats
from sklearn.model_selection import train_test_split
from np_to_tfrecords import np_to_tfrecords
import pywt
__author__ = "Diego Cabrera"
__copyright__ = "Copyright 2018, The GIDTEC Fault Diagnosis Project"
__credits__ = ["Diego Cabrera"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Diego Cabrera"
__email__ = "dcabrera@ups.edu.ec"
__status__ = "Prototype"


def signal2wp_energy(signal, wavelets, max_level):
    signal = signal.squeeze()
    energy_coef = np.zeros((len(wavelets),2**max_level))
    for j, wavelet in enumerate(wavelets):
        wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
        level = wp_tree.get_level(max_level, order='freq')
        for i, node in enumerate(level):
            energy_coef[j,i] = np.sqrt(np.sum(node.data ** 2)) / node.data.shape[0]
            # print(i)
            # print(node.data.shape[0])
            # print('next----------')
    return energy_coef.flatten()

def signal2wp_spectrum(signal, wavelet, max_level):
    signal = signal.squeeze()
    wp_tree = pywt.WaveletPacket(data=signal, wavelet=wavelet, maxlevel=max_level)
    nodes = wp_tree.get_level(max_level, order='freq')
    spectrum = np.array([np.abs(n.data) for n in nodes])
    spectrum = np.log(spectrum+0.1)
    mean = spectrum.mean()
    spectrum = spectrum - mean
    max = np.abs(spectrum).max()
    spectrum = spectrum / max

    return spectrum.flatten()


def rectified_average(signal):
    """
    Compute rectified average feature
    :param signal: time-series
    :type signal: numpy array
    :return: rectified average of signal
    :rtype: float
    """
    return np.mean(np.abs(signal))


def statistic_features(signal):
    """
    Compute a group of statistical features
    :param signal: time-series
    :type signal: numpy array
    :return: group of statistical feature from the signal
    :rtype: tuple
    """
    mean = np.mean(signal)
    rms = np.sqrt(np.mean(np.square(signal)))
    std_dev = np.std(signal)
    kurtosis = scipy.stats.kurtosis(signal)
    peak = np.max(signal)
    crest = peak / rms
    r_mean = rectified_average(signal)
    form = rms / r_mean
    impulse = peak / r_mean
    variance = std_dev ** 2
    minimum = np.min(signal)
    return mean, rms, std_dev, kurtosis, peak, crest, r_mean, form, impulse, variance, minimum

def build_subsets_labels(labels):
    number = [int(re.split('R|P|.mat', file)[-2]) for file in labels]
    number = np.array(number,dtype=int)
    labels = np.array(labels)
    #labels = labels[number <= 2]
    #number = number[number <= 2]

    files_train,files_tmp,_,number_tmp =train_test_split(labels,number,train_size=0.7,stratify=number)
    files_val, files_test, _, _ = train_test_split(files_tmp, number_tmp, train_size=0.5,stratify=number_tmp)
    return {'train':files_train, 'val':files_val, 'test':files_test}


def processing_dataset_features(path, path_out, standarize=False):
    """
    Method to process the dataset in parallel
    :param path: path to raw signals dataset
    :type path: string
    :return: features dataset
    :rtype: numpy array
    """
    files = []
    labels = []
    for dir in os.listdir(path):
        folder = join(path, dir)
        list_files = [join(folder, file) for file in os.listdir(folder)]
        labels.extend(os.listdir(folder))
        files.extend(list_files)
    sets = build_subsets_labels(files)

    for subset in ['train','val','test']:
        features = []
        labels = []
        p = Pool()
        iterador = p.imap(processing_file_features, sets[subset])
        for iteracion in iterador:
            feature, label = iteracion
            features.extend(feature)
            labels.extend(label)
        features = np.array(features,dtype=np.float32).squeeze()
        labels = np.array(labels).squeeze()
        print(features.shape, labels.shape)

        if standarize:
            if subset == 'train':
                average = np.mean(features,axis=0)
                standard = np.std(features, axis=0)
            features = (features - average) / standard

        np_to_tfrecords(features,labels,path_out+'/'+subset+'_set')
        print('Finish building subset ' + subset)
        if subset == 'train':
            upper_fault = np.max(np.unique(labels[:, -1]))
            f = open(path_out+'/'+'log','w')
            for fault in range(2,upper_fault+1):
                indexes = np.where(labels[:, -1] == fault)
                indexes = indexes[0]
                np.random.shuffle(indexes)
                length = indexes.shape[0]
                for portion in [1]:
                    upper = int(portion*length/100)
                    f.write('fault ' + str(fault) + ',' + str(portion) + '%' + ':' + str(upper) + ' samples\n')
                    print('fault', fault ,',', portion,'%',':',upper,'samples')
                    np_to_tfrecords(features[indexes[:upper]],
                                    labels[indexes[:upper]],
                                    path_out + '/' + 'class' + str(fault) + '_' + str(portion) + '_set')
            f.close()
        p.close()


def processing_dataset_tf(path, path_out):
    """
    Method to process the dataset in parallel
    :param path: path to raw signals dataset
    :type path: string
    :return: features dataset
    :rtype: numpy array
    """
    files = []
    labels = []
    for dir in os.listdir(path):
        folder = join(path, dir)
        list_files = [join(folder, file) for file in os.listdir(folder)]
        labels.extend(os.listdir(folder))
        files.extend(list_files)
    sets = build_subsets_labels(files)

    for subset in ['train','val','test']:
        features = []
        labels = []
        p = Pool()
        iterador = p.imap(processing_file_tf, sets[subset])
        for iteracion in iterador:
            feature, label = iteracion
            features.extend(feature)
            labels.extend(label)
        features = np.array(features,dtype=np.float32).squeeze()
        '''
        if subset == 'train':
            minimum = features.min()
            features = features - minimum
            medium = features.max()/2
            features = (features - medium) / medium
        else:
            features = features - minimum
            features = (features - medium) / medium
        '''
        print(subset,'max:',features.max())
        print(subset,'min:',features.min())
        labels = np.array(labels).squeeze()
        print(features.shape)
        features = features.reshape((-1,features.shape[1]*features.shape[2]))
        print(features.shape, labels.shape)
        np_to_tfrecords(features,labels,path_out+'/'+subset+'_set')
        print('Finish building subset ' + subset)
        if subset == 'val':
            np_to_tfrecords(features[labels[:,1] == 2], labels[labels[:,1] == 2], path_out + '/' + 'class2' + '_set')
        p.close()

def processing_file_features(file):
    """
    Method to process a raw signal file
    :param file: name of raw signal .mat file
    :type file: string
    :return: group of features for the signal in .mat file
    :rtype: numpy array
    """
    length = 10000
    step = 200
    label = [int(x) for x in re.split('R|P|.mat', file)[-3:-1]]
    print(file)
    data = sio.loadmat(file)
    signal = data['data']['Analog50k'][0][0][0][0][5]

    i = 0
    signals = []
    while i + length <= signal.shape[0]:
        signals.append(signal[i:i+length])
        i += step

    signals = np.array(signals)
    features = [signal2wp_energy(chunk,['db7','sym3'],6) for chunk in signals]
    features = np.array(features)
    labels = np.repeat([label],signals.shape[0],axis=0)
    return features, labels

def processing_file_tf(file):
    """
    Method to process a raw signal file
    :param file: name of raw signal .mat file
    :type file: string
    :return: group of features for the signal in .mat file
    :rtype: numpy array
    """
    length = 3328
    step = 200
    label = [int(x) for x in re.split('R|P|.mat', file)[-3:-1]]
    print(file)
    data = sio.loadmat(file)
    signal = data['data']['Analog50k'][0][0][0][0][5]

    i = 0
    signals = []
    while i + length <= signal.shape[0]:
        signals.append(signal[i:i+length])
        i += step

    signals = np.array(signals)
    features = [signal2wp_spectrum(chunk,'db7',6) for chunk in signals]
    features = np.array(features)
    labels = np.repeat([label],signals.shape[0],axis=0)
    return features, labels

if __name__ == "__main__":
    path = '/home/titan/databases/RawData/Raw_Data_Valve_Faults_DB_011V0/'
    path_out = 'datasetValveCompressor'

    if not os.path.exists('../data/' + path_out):
        os.makedirs('../data/' + path_out)
    # Datasets generation process
    processing_dataset_features(path, '../data/' + path_out)