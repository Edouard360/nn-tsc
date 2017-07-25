# b3 model - vgg16 for 1 d

## Model parameters

n_hidden = np.array([16,32,64,128,128])
kernel_size = np.array([9,9,9,9,9])
pool_strides = np.array([4,4,4,4,4])
But only 3 batches...
Then Dense 100
Dropout 0.3
Dense num_classes

Was stopped because requires *64 layer size


## Main parameters

- epochs : 3000
- batch_size = 32
- EarlyStopping(monitor='loss', patience=1000, min_delta=1e-5)
- ReduceLROnPlateau(monitor='loss', factor=0.5,patience=50, min_lr=1e-6)


