# b1 model

## Model parameters

Two convolutions with batch norm and relu. 
16 - 5
14 - 4
Dropout
2 fully connected layers - (40) then num_classes.

## Main parameters

- epochs : 3000
- EarlyStopping(monitor='loss', patience=1000, min_delta=1e-5)
- ReduceLROnPlateau(monitor='loss', factor=0.5,patience=50, min_lr=1e-6)
