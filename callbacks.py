import keras


class SoftVerbose(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.epochs = self.params['epochs']

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 10 == 0):
            print("Loss: %5.3f - Acc: %5.3f (%i,%i)" % (logs['loss'], logs['acc'], epoch, self.epochs))
        if (epoch % 100 == 0):
            print("\nValidation\nLoss: %5.3f - Acc: %5.3f\n" % (logs['val_loss'], logs['val_acc']))
