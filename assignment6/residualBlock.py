import tensorflow as tf
import tensorflow.keras as K

class ResidualBlock(K.layers.Layer):

    def __init__(self, input_channel_dimension):
        super(ResidualBlock, self).__init__()
        self.conv1 = K.layers.Conv2D(
            filters = 8, kernel_size = 1,
            padding = "same", strides = 1,
            activation = K.activations.relu
            )
        self.batchnorm1 = K.layers.BatchNormalization()
        self.conv2 = K.layers.Conv2D(
            filters=8, kernel_size=3,
            padding="same", strides=1,
            activation=K.activations.relu
        )
        self.batchnorm2 = K.layers.BatchNormalization()
        self.conv3 = K.layers.Conv2D(
            filters=input_channel_dimension, kernel_size=1,
            padding="same", strides=1,
            activation=K.activations.relu
        )


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.conv3(x)
        x = tf.add(x, inputs)
        return x