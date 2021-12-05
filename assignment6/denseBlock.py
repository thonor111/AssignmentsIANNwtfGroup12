import tensorflow as tf
import tensorflow.keras as K

class DenseBlock(K.layers.Layer):

    def __init__(self):
        super(DenseBlock, self).__init__()
        self.conv1 = K.layers.Conv2D(
            filters=128, kernel_size=1,
            padding="same", strides=1
        )
        self.batchnorm1 = K.layers.BatchNormalization()
        self.activation = K.layers.Activation("relu")
        self.conv2 = K.layers.Conv2D(
            filters=32, kernel_size=3,
            padding="same", strides=1
        )
        self.batchnorm2 = K.layers.BatchNormalization()
        self.conv3 = K.layers.Conv2D(
            filters=32, kernel_size=1,
            padding="same", strides=1
        )
        self.batchnorm3 = K.layers.BatchNormalization()


    def call(self, inputs):
        x = self.batchnorm1(inputs)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = K.layers.concatenate([x, inputs])
        return x