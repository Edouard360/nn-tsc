#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: edouard
"""

import keras
import numpy as np
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras.layers import Concatenate, Conv1D, MaxPool1D, Activation, Dense
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.utils import plot_model

from callbacks import SoftVerbose
from layers import ConvDiff, AutoReshape
from tools import train_test_ucr, initialize_dataframe, get_ucr_list

# import tensorflow as tf
# from keras.backend.tensorflow_backend import set_session
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.3
# config.gpu_options.allow_growth=True
# set_session(tf.Session(config=config))

fdir = "./ucr/"
flist = get_ucr_list()

soft_verbose = SoftVerbose()
# tensorboard = TensorBoard(log_dir = './logs',write_graph=True,write_images=True)
dataframe = initialize_dataframe("logs/results.csv")

epochs = 10

for each in flist:
    fname = each
    x_train, Y_train, x_test, Y_test, nb_classes = train_test_ucr(fdir, fname)
    batch_size = max(min(int(x_train.shape[0] / 10), 16), 64)
    x = keras.layers.Input(x_train.shape[1:])
    #    drop_out = Dropout(0.2)(x)

    x_diff = ConvDiff()(x)
    x_combined = Concatenate(axis=2)([x_diff, x])
    conv1 = Conv1D(16, 8, padding='same')(x_combined)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('relu')(conv1)

    #    drop_out = Dropout(0.2)(conv1)
    conv2 = Conv1D(32, (5), padding='same')(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('relu')(conv2)

    #    drop_out = Dropout(0.2)(conv2)
    conv3 = Conv1D(16, (3), padding='same')(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('relu')(conv3)
    # full = keras.layers.GlobalMaxPooling2D()(conv3)

    full = keras.layers.pooling.MaxPool1D(pool_size=5, strides=5, padding="valid")(conv3)
    full = AutoReshape()(full)

    out = Dense(nb_classes, activation='softmax')(full)

    model = Model(inputs=x, outputs=out)

    optimizer = keras.optimizers.Adam()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    # model.load_weights('./models/' + fname + '.h5')
    # json_string = model.to_json()
    # plot_model(model, to_file='./model.png')

    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5,
                                  patience=50, min_lr=0.0001)
    hist = model.fit(x_train, Y_train, batch_size=batch_size, epochs=epochs,
                     verbose=0, validation_data=(x_test, Y_test), callbacks=[reduce_lr, soft_verbose])  # ,tensorboard])

    # model.save_weights('./models/' + fname + '.h5')
    # Print the testing results which has the lowest training loss.

    log = pd.DataFrame(hist.history)
    best_val_acc = np.round(log.loc[log['val_acc'].idxmax]['val_acc'], 2)
    final_val_acc = np.round(log.loc[log['loss'].idxmin]['val_acc'], 2)
    dataframe.ix[fname, :] = best_val_acc, final_val_acc
    print("For dataset %s\nBest test accuracy: %.2f\nFinal accuracy: %.2f\n" % (fname, best_val_acc, final_val_acc))
    dataframe.to_csv(dataframe.path)
