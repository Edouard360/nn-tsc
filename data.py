import numpy as np
from keras.utils import np_utils

def data():
    def readucr(filename):
        data = np.loadtxt(filename, delimiter=',')
        Y = data[:, 0]
        X = data[:, 1:]
        return X, Y

    def to_categorical(y, nb_classes):
        return np_utils.to_categorical((y - y.min()) / (y.max() - y.min()) * (nb_classes - 1), nb_classes)

    fdir = "./ucr/"  # Path to the UCR Time Series Data directory
    fname = "ChlorineConcentration"
    print("Dataset : " + fname)

    x_train, y_train = readucr(fdir + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fdir + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    Y_train = to_categorical(y_train, nb_classes)
    Y_test = to_categorical(y_test, nb_classes)
    return x_train, x_test, Y_train, Y_test