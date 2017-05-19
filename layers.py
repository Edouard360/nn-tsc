import tensorflow as tf
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers.core import Reshape
from keras.utils import conv_utils


class FFT(Layer):
    def call(self, inputs):
        casted = tf.cast(inputs, tf.complex64)
        changed = tf.spectral.fft(casted)
        changed2 = tf.abs(changed)
        return tf.cast(changed2, tf.float32)


class SlidingVariance(Layer):
    def __init__(self, w, **kwargs):
        self.w = w
        super(SlidingVariance, self).__init__(**kwargs)

    def build(self, input_shape):
        self.len = input_shape[1]
        super(SlidingVariance, self).build(input_shape)

    def call(self, inputs):
        def var(i):
            result = tf.cond(i + self.w < self.len, lambda: tf.slice(inputs, [0, i, 0], [-1, self.w, 1]),
                             lambda: tf.slice(inputs, [0, i, 0], [-1, -1, 1]))
            mean, variance = tf.nn.moments(result, axes=[1])
            return variance

        mapping = tf.map_fn(var, tf.range(int(self.len), dtype=tf.int32), dtype=tf.float32)
        sliding_var = tf.concat(mapping, axis=1)
        return sliding_var


class ConvDiff(Layer):
    def __init__(self, **kwargs):
        super(ConvDiff, self).__init__(**kwargs)

    def build(self, input_shape):
        diff_init = lambda shape: K.constant([1, -1], shape=shape)
        self.kernel = self.add_weight(shape=(2, 1, 1), name="diff_weights",
                                      initializer=diff_init,
                                      trainable=False)
        super(ConvDiff, self).build(input_shape)

    def call(self, inputs):
        outputs = K.conv1d(
            inputs,
            self.kernel,
            strides=1,
            padding='same',
            data_format="channels_last")
        return outputs

    def compute_output_shape(self, input_shape):
        new_dim = conv_utils.conv_output_length(input_shape[1], filter_size=2, stride=1, padding='same')
        return (input_shape[0], new_dim, 1)


class AutoReshape(Layer):
    def __init__(self, **kwargs):
        self.n_features = 1
        super(AutoReshape, self).__init__(**kwargs)

    def build(self, input_shape):
        for shape in input_shape[1:]:
            self.n_features = self.n_features * shape
        super(AutoReshape, self).build(input_shape)

    def call(self, inputs):
        return Reshape((self.n_features,))(inputs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_features)
