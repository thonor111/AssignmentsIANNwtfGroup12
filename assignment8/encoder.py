'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K


class Encoder(K.Model):

    def __init__(self):
        '''
        Initializes the encoder
        '''
        super(Encoder, self).__init__()
        self.conv_1 = K.layers.Conv2D(filters = 8, kernel_size = 3, strides = 1, padding = "same", activation = "relu")
        self.conv_2 = K.layers.Conv2D(filters = 8, kernel_size = 3, strides = 2, padding = "same", activation = "relu")
        self.conv_3 = K.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")
        self.conv_4 = K.layers.Conv2D(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")
        self.conv_5 = K.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")
        self.conv_6 = K.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")
        self.flatten = K.layers.Flatten()

    @tf.function
    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.flatten(x)
        return x
