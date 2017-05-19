import numpy as np
from keras.utils import np_utils


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y

def to_categorical(y,nb_classes):
    return np_utils.to_categorical((y - y.min()) / (y.max() - y.min()) * (nb_classes - 1), nb_classes)

def train_test_ucr(fdir,fname):
    x_train, y_train = readucr(fdir + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fdir + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    Y_train = to_categorical(y_train,nb_classes)
    Y_test = to_categorical(y_test,nb_classes)

    x_train_mean = x_train.mean()
    x_train_std = x_train.std()
    x_train = (x_train - x_train_mean) / (x_train_std)

    x_test = (x_test - x_train_mean) / (x_train_std)
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train,Y_train,x_test,Y_test,nb_classes