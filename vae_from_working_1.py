'''This script demonstrates how to build a variational autoencoder
with Keras and deconvolution layers.
Reference: "Auto-Encoding Variational Bayes" https://arxiv.org/abs/1312.6114
'''
import numpy as np
import matplotlib.pyplot as plt

from keras.layers import Input, Dense, Lambda, Flatten, Reshape, MaxPooling1D, Conv1D, UpSampling1D
from keras.models import Model
from keras import backend as K
from keras import metrics

from sklearn.preprocessing import MinMaxScaler
from tools import readucr, to_categorical

fdir = "./ucr/"
fname = "ChlorineConcentration"
print("Dataset : "+fname)

x_train, y_train  = readucr(fdir + fname + '/' + fname + '_TRAIN')
x_test, y_test  =readucr(fdir + fname + '/' + fname + '_TEST')
nb_classes = len(np.unique(y_test))
Y_train = to_categorical(y_train, nb_classes)
Y_test = to_categorical(y_test, nb_classes)

print("Shape : "+str(x_train.shape[1]))
print("Nb classes : "+str(nb_classes))
print("Train size : "+str(x_train.shape[0]))
print("Test size : "+str(x_test.shape[0]))

scale = MinMaxScaler()
x_concat = np.concatenate([x_train,x_test])
x_concat = scale.fit_transform(x_concat)
x_train = x_concat[:x_train.shape[0]]
x_test = x_concat[x_train.shape[0]:]

crop_length = 160
x_train = x_train[:,:crop_length]
x_test = x_test[:,:crop_length]
x_concat = x_concat[:,:crop_length]

x_train = x_train.reshape(x_train.shape+(1,))
x_test = x_test.reshape(x_test.shape+(1,))
x_concat = x_concat.reshape(x_concat.shape+(1,))
print(x_concat.shape)

x_concat = x_concat[:3000] # BE CAREFUL !! The size must be a correct multiple of the batch_size
epochs = 100



x = Input(shape=x_train.shape[1:])
depth = 32
kernel_size = 3
intermediate_dim = 128
latent_dim = 10
epsilon_std = 1.0
batch_size = 100 # Be extremely careful ! This
print("Batch size is : "+str(batch_size))
output_shape = (batch_size, ((crop_length//2)//2), depth)
# (channels last with tensorflow) / with theano, channels first (batch_size, depth, ((crop_length//2)//2))

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size,latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon


# The network
h = x
h = Conv1D(depth, kernel_size, activation='relu', padding='same')(h) #name="test" # NO ! We duplicate
h = MaxPooling1D(2, padding='same')(h)
h = Conv1D(depth,kernel_size, activation='relu', padding='same')(h)
h = MaxPooling1D(2, padding='same')(h)

flat = Flatten()(h)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

y = z
y = Dense(intermediate_dim, activation='relu')(y)
y = Dense(depth * ((crop_length//2)//2), activation='relu')(y)
y = Reshape(output_shape[1:])(y)
y = Conv1D(depth, kernel_size, activation='relu', padding='same')(y)
y = UpSampling1D(2)(y)
y = Conv1D(depth, kernel_size, activation='relu', padding='same')(y)
y = UpSampling1D(2)(y)
y = Conv1D(1, kernel_size, activation='sigmoid', padding='same')(y)

def vae_loss(x, y):
    x = K.flatten(x) # Flattening is essential apparently
    y = K.flatten(y)
    xent_loss = crop_length * metrics.binary_crossentropy(x, y)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return K.mean(xent_loss + kl_loss)

vae = Model(x, y)
vae.compile(optimizer='rmsprop', loss=vae_loss)
#vae.summary()

vae.fit(x_concat, x_concat, shuffle=True, epochs=epochs, batch_size=batch_size, validation_data=(x_concat, x_concat))

# build a model to project inputs on the latent space
encoder = Model(x, z_mean)

# display a 2D plot of the digit classes in the latent space
x_test_encoded = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(x_test_encoded[:, 0], x_test_encoded[:, 1], c=y_test)
plt.colorbar()
plt.show()

generator = Model(z, y)