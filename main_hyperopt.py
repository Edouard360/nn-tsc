import pickle
import sys, traceback

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.layers import MaxPooling1D, Input, Dense, Dropout, Conv1D, Reshape, Flatten, UpSampling1D, \
    BatchNormalization, Activation
from keras.models import Model

from data import data
from mail import sendMessageException

x_train, x_test, Y_train, Y_test = data()

space = {
    'n_layers': hp.choice('n_layers',
                          [{'n':1},
                           {'n':2,'squeeze':hp.choice('squeeze',[1,2])}]),
    'after_conv': hp.choice('after_conv',
                            [{'type':'max_pooling',
                              'activation':hp.choice('activation',['relu','elu',None]),
                              'pooling_size':hp.choice('pooling_size',[2,4])},
                             {'type':'batch_norm',
                              'activation': 'relu',
                              'pooling_size':None}]),
    'depth': hp.choice('depth',[16,32,64]),
    'kernel_size': hp.choice('kernel_size',[4,8,16]),
    'dropout': hp.uniform('dropout',.0,.5),
    'intermediate_dim': hp.choice('intermediate_dim',[16,32,64]),
    'recons_regul': hp.uniform('recons_regul',1.,10.),
    'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
}

def f_nn(params):
    crop_length = 160

    x_train_ = x_train[:, :crop_length]
    x_test_ = x_test[:, :crop_length]

    x_concat = np.concatenate([x_train_, x_test_])
    x_concat = (x_concat - x_concat.min()) / (x_concat.max() - x_concat.min()) * 2 - 1.
    x_train_ = x_concat[:x_train_.shape[0]]
    x_test_ = x_concat[x_train_.shape[0]:]

    x_train_ = x_train_.reshape(x_train_.shape + (1,))
    x_test_ = x_test_.reshape(x_test_.shape + (1,))
    x_concat = np.concatenate([x_train_, x_test_],axis = 0)

    epochs = 1000
    batch_size = 256

    depth = params['depth']
    n_layers = params['n_layers']['n']
    kernel_size = params['kernel_size']
    intermediate_dim = 32

    if (n_layers == 2):
        squeeze = params['n_layers']['squeeze']

    after_conv_type = params['after_conv']['type']
    if(after_conv_type=='max_pooling'):
        AfterConvLayer = lambda: lambda h:MaxPooling1D(params['after_conv']['pooling_size'], padding='same')(h)
    elif(after_conv_type=='batch_norm'):
        AfterConvLayer = BatchNormalization # if BatchNormalization() this will result in the same instanciation being used twice

    activation = params['after_conv']['activation']
    pooling = params['after_conv']['pooling_size']

    undersample_rate = 1 if pooling is None else pooling**(params['n_layers']['n'])
    output_shape = (batch_size,crop_length//undersample_rate, 1)  # //4 //4

    # use_biais = True / False
    # elu / selu / relu
    # dilation_rate -> but stride = 1

    x = Input(shape=x_train_.shape[1:])
    h = x
    h = Conv1D(depth, kernel_size, padding='same')(h)  # name='test' # NO ! We duplicate
    h = AfterConvLayer()(h)
    h = Activation(activation)(h)

    if (n_layers == 2):
        h = Conv1D(depth * squeeze,kernel_size // squeeze, padding='same', activation=activation)(h)
        h = AfterConvLayer()(h)
        h = Activation(activation)(h)


    flat = Flatten()(h)
    hidden = Dense(intermediate_dim, activation=activation)(flat)

    x_recons = hidden
    x_recons = Dense(output_shape[1])(x_recons)
    x_recons = Reshape(output_shape[1:])(x_recons)

    if(n_layers == 2):
        x_recons = Conv1D(depth * squeeze, kernel_size // squeeze, padding='same', activation=activation)(x_recons)
        if(after_conv_type=='max_pooling'):
            x_recons = UpSampling1D(pooling)(x_recons)

    x_recons = Conv1D(depth, kernel_size, padding='same',activation=activation)(x_recons)
    if(after_conv_type=='max_pooling'):
        x_recons = UpSampling1D(pooling)(x_recons)

    x_recons = Conv1D(1, kernel_size, padding='same')(x_recons)

    y = Dropout(0.2)(hidden)
    y = Dense(Y_train.shape[1], activation='softmax')(y)

    mlt = Model(x, [x_recons, y])
    mlt.compile(params['optimizer'], ['mse', 'categorical_crossentropy'], metrics=['acc'])

    hist = mlt.fit(x_concat, [x_concat, np.concatenate([Y_train, Y_test])],
                   sample_weight=[params['recons_regul']*np.ones(len(x_concat)),
                                  np.concatenate([np.ones(len(Y_train)),
                                                  np.zeros(len(Y_test))])],
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=0
                   )

    acc = (np.argmax(mlt.predict(x_test_)[1], axis=1) == np.argmax(Y_test, axis=-1)).mean()
    print('ACC :', acc)
    return {'loss': -acc, 'status': STATUS_OK} # we could add the history...


trials = Trials()

def run_trials():

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open('logs/opt/trials.p', 'rb'))
        max_trials = len(trials.trials) + trials_step
        print('Found saved Trials!')
    except:  # create a new trials object and start searching
        trials = Trials()

    try:
        best = fmin(f_nn,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=max_trials,
                    trials=trials,
                    )

        # save the trials object
        with open('logs/opt/trials.p', 'wb') as test:
            pickle.dump(trials, test)

        with open('logs/opt/space.p', 'wb') as test:
            pickle.dump(space, test)
    except:
        sendMessageException(traceback.format_exc())




# loop indefinitely and stop whenever you like
while True:
    run_trials()