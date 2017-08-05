import numpy as np
import pandas as pd
from keras.regularizers import l1, l2
from keras.utils import np_utils
import pickle
from tools import initialize_dataframe
from data import data

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice, uniform, conditional
from keras.layers import MaxPooling1D, Input, Dense, Dropout, Conv1D, Reshape, Flatten, UpSampling1D,BatchNormalization, Activation
from keras.models import Model

def model(x_train, x_test, Y_train, Y_test):
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

    epochs = 3
    batch_size = 128

    depth = {{choice([16])}}
    n_layers = {{choice(["1 layer", "2 layers"])}}

    if (conditional(n_layers) == "2 layers"):
        squeeze = {{choice([1,2])}}

    layer_after_conv = {{choice(["batch_norm","max_pool"])}}

    if(conditional(layer_after_conv)=="batch_norm"):
        activation = "relu"  # "Batchnorm + relu","maxPooling"
        pooling = 1
    elif(conditional(layer_after_conv)=="max_pool"):
        activation = {{choice(["relu","elu",None])}} # "Batchnorm + relu","maxPooling"
        pooling = {{choice([2, 4])}}

    kernel_size = 8
    intermediate_dim = {{choice([32])}}

    undersample_rate = pooling**(2 if (conditional(n_layers)=="2 layers") else 1)
    output_shape = (batch_size,crop_length//undersample_rate, 1)  # //4 //4

    # use_biais = True / False
    # elu / selu / relu
    # dilation_rate -> but stride = 1

    x = Input(shape=x_train_.shape[1:])
    h = x
    h = Conv1D(depth, kernel_size, padding='same')(h)  # name="test" # NO ! We duplicate
    if (conditional(layer_after_conv) == "max_pool"):
        h = MaxPooling1D(pooling, padding='same')(h)
    else:
        h = BatchNormalization()(h)
    h = Activation(activation)(h)

    if(conditional(n_layers)=="2 layers"):
        h = Conv1D(depth * squeeze,kernel_size // squeeze, padding='same', activation=activation)(h)
        if (conditional(layer_after_conv) == "max_pool"):
            h = MaxPooling1D(pooling, padding='same')(h)
        else:
            h = BatchNormalization()(h)
        h = Activation(activation)(h)

    flat = Flatten()(h)
    hidden = Dense(intermediate_dim, activation=activation)(flat)

    x_recons = hidden
    x_recons = Dense(output_shape[1])(x_recons)
    x_recons = Reshape(output_shape[1:])(x_recons)

    if(conditional(n_layers)=="2 layers"):
        x_recons = Conv1D(depth * squeeze, kernel_size // squeeze, padding='same', activation=activation)(x_recons)
        if (conditional(layer_after_conv) == "max_pool"):
            x_recons = UpSampling1D(pooling)(x_recons)

    x_recons = Conv1D(depth, kernel_size, padding='same',activation=activation)(x_recons)
    if (conditional(layer_after_conv) == "max_pool"):
        x_recons = UpSampling1D(pooling)(x_recons)
    x_recons = Conv1D(1, kernel_size, padding='same')(x_recons)

    y = Dropout(0.2)(hidden)
    y = Dense(Y_train.shape[1], activation='softmax')(y)

    mlt = Model(x, [x_recons, y])
    mlt.compile('adam', ['mse', 'categorical_crossentropy'], metrics=['acc'])

    hist = mlt.fit(x_concat, [x_concat, np.concatenate([Y_train, Y_test])],
                   sample_weight=[np.ones(len(x_concat)),
                                  np.concatenate([np.ones(len(Y_train)),
                                                  np.zeros(len(Y_test))])],
                   shuffle=True,
                   epochs=epochs,
                   batch_size=batch_size,
                   verbose=0
                   )
    acc = (np.argmax(mlt.predict(x_test_)[1], axis=1) == np.argmax(Y_test, axis=-1)).mean()
    print("ACC :", acc)
    return {'loss': -acc, 'status': STATUS_OK} # we could add the history...


trials = Trials()

def run_trials():

    trials_step = 3  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 5
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open("logs/opt/trials.p", "rb"))
        max_trials = len(trials.trials) + trials_step
        print("Found saved Trials!")
    except:  # create a new trials object and start searching
        trials = Trials()

    best_run, best_model, space = optim.minimize(model=model,
                                                 data=data,
                                                 algo=tpe.suggest,
                                                 max_evals=max_trials,
                                                 trials=trials,
                                                 eval_space=True,
                                                 return_space=True)

    # save the trials object
    with open("logs/opt/trials.p", "wb") as test:
        pickle.dump(trials, test)

    with open("logs/opt/space.p", "wb") as test:
        pickle.dump(space, test)


# loop indefinitely and stop whenever you like
while True:
    run_trials()

print("Best performing model chosen hyper-parameters:")
print(best_run)