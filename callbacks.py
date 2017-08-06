import keras

class Verbose1(keras.callbacks.Callback):
    def __init__(self, loss_frequency=10, validation_frequency=100):
        super(Verbose1, self).__init__()
        self.loss_frequency = loss_frequency
        self.validation_frequency = validation_frequency

    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.loss_frequency == 0):
            print("Loss: %5.3f - Acc: %5.3f (%i,%i)" % (logs['loss'], logs['dense_4_acc'], epoch, self.epochs))
        # if (epoch % self.validation_frequency == 0):
        #     print("\nValidation\nLoss: %5.3f - Acc: %5.3f\n" % (logs['val_loss'], logs['val_acc']))


class Verbose2(keras.callbacks.Callback):
    def __init__(self, loss_frequency=10, validation_frequency=100):
        super(Verbose2, self).__init__()
        self.loss_frequency = loss_frequency
        self.validation_frequency = validation_frequency

    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % self.loss_frequency == 0):
            print("Loss: %5.3f (%i,%i)" % (logs['loss'], epoch, self.epochs))
        if (epoch % self.validation_frequency == 0):
            print("\nValidation\nLoss: %5.3f\n" % (logs['val_loss']))
