import pickle
import sys, traceback

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from keras.engine import Layer
from keras.layers import MaxPooling1D, Input, Dense, Dropout, Conv1D, Reshape, Flatten, UpSampling1D, \
    BatchNormalization, Activation, Lambda
from keras.models import Model
from keras.losses import mse,categorical_crossentropy
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
    'middle_layer': hp.choice('middle_layer',
                              [{'type':'gaussian',
                               'epsilon':hp.choice('epsilon',[0.1,0.5,1]),
                               'correct_factor': hp.choice('correct_factor',[True, False]),
                               'gaussian_regul':hp.choice('gaussian_regul',[0.1,0.5,5,10])},
                               {'type':'regular',
                                'epsilon':None,
                                'correct_factor':None,
                                'gaussian_regul':None}
                              ]),
    'depth': hp.choice('depth',[16,32,64]),
    'kernel_size': hp.choice('kernel_size',[2,4,8,16]),
    'intermediate_dim': hp.choice('intermediate_dim',[16,32,64]),
    'dropout': hp.uniform('dropout',.0,.5),
    'recons_regul': hp.uniform('recons_regul',1.,10.),
    'optimizer': hp.choice('optimizer',['adadelta','adam','rmsprop']),
}
from keras import backend as K

def sampling(intermediate_dim,epsilon_std):
    def lambda_sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(intermediate_dim,), # better than (batch_size, latent_dim) since batch_size can vary !
                                  mean=0., stddev=epsilon_std)
        return z_mean + K.exp(z_log_var) * epsilon
    return lambda_sampling

class CustomVariationalLayer(Layer):
    def __init__(self, flatten_length,loss_factor = 0.5, epsilon_std = 0.1,correct_factor = False, **kwargs):
        self.flatten_length = flatten_length
        self.epsilon_std = epsilon_std
        self.loss_factor = loss_factor
        self.correct_factor = correct_factor
        super(CustomVariationalLayer, self).__init__(**kwargs)

    def gaussian_loss(self, hidden_mean, hidden_log_var):
        correction_factor = self.epsilon_std if self.correct_factor else 1
        return K.mean(- self.loss_factor * K.sum(1 + hidden_log_var - K.square(hidden_mean) - correction_factor * K.exp(hidden_log_var), axis=-1))

    def call(self, hidden_mean_and_log_var):
        hidden_mean = hidden_mean_and_log_var[0]
        hidden_log_var = hidden_mean_and_log_var[1]
        epsilon_std = self.epsilon_std
        loss = self.gaussian_loss(hidden_mean, hidden_log_var)
        self.add_loss(loss)
        epsilon = K.random_normal(shape=(self.flatten_length,), # better than (batch_size, latent_dim) since batch_size can vary !
                                  mean=0., stddev=epsilon_std)
        return hidden_mean + K.exp(hidden_log_var) * epsilon


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


    intermediate_dim = params['intermediate_dim']
    middle_layer = params['middle_layer']['type']

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
    if(middle_layer == 'gaussian'):
        flat = Dense(intermediate_dim, activation=activation)(flat)
        hidden_mean = Dense(intermediate_dim, activation=activation)(flat)
        hidden_log_var = Dense(intermediate_dim, activation=activation)(flat)
        #hidden = Lambda(sampling(intermediate_dim,epsilon_std), output_shape=(intermediate_dim,))([hidden_mean, hidden_log_var])
        hidden = CustomVariationalLayer(intermediate_dim,
                                        epsilon_std=params['middle_layer']['epsilon'],
                                        loss_factor=params['middle_layer']['gaussian_regul'],
                                        correct_factor=params['middle_layer']['correct_factor'])([hidden_mean, hidden_log_var])
    elif (middle_layer == 'regular'):
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

    y = Dropout(params['dropout'])(hidden)
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
    except KeyboardInterrupt:
        raise "Process stopped"
    except:
        sendMessageException(traceback.format_exc())


# loop indefinitely and stop whenever you like
while True:
    run_trials()