import os

import numpy as np
import pandas as pd
from keras.utils import np_utils


def get_ucr_list(path="./ucr"):
    list_datasets = os.listdir(path)
    assert len(list_datasets) == 85
    return list_datasets


def readucr(filename):
    data = np.loadtxt(filename, delimiter=',')
    Y = data[:, 0]
    X = data[:, 1:]
    return X, Y


def to_categorical(y, nb_classes):
    return np_utils.to_categorical((y - y.min()) / (y.max() - y.min()) * (nb_classes - 1), nb_classes)


def train_test_ucr(fdir, fname, reshape=True, scale = False, categorical = True):
    x_train, y_train = readucr(fdir + fname + '/' + fname + '_TRAIN')
    x_test, y_test = readucr(fdir + fname + '/' + fname + '_TEST')
    nb_classes = len(np.unique(y_test))
    if categorical:
        Y_train = to_categorical(y_train, nb_classes)
        Y_test = to_categorical(y_test, nb_classes)
    else:
        Y_train = y_train
        Y_test = y_test
    if not scale:
        x_train_mean = x_train.mean()
        x_train_std = x_train.std()
        x_train = (x_train - x_train_mean) / (x_train_std)
        x_test = (x_test - x_train_mean) / (x_train_std)
    else:
        train_min = min(x_train.min(), x_test.min())
        train_max = max(x_train.max(), x_test.max())
        x_train = (x_train-train_min)/(train_max-train_min)
        x_test = (x_test-train_min)/(train_max-train_min)
    if reshape:
        x_train = x_train.reshape(x_train.shape + (1,))
        x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, Y_train, x_test, Y_test, nb_classes


def initialize_dataframe(path, features = ["best_val_acc", "final_val_acc","stopped_epoch"], rows = None):
    if (os.path.isfile(path)):
        dataframe = pd.read_csv(path, index_col=0)
    else:
        datasets_list = get_ucr_list() if rows is None else rows
        features = features
        dataframe = pd.DataFrame(0, index=datasets_list, columns=features)
        dataframe.to_csv(path)
    dataframe.path = path
    return dataframe
