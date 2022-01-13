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
        self.conv_1 = K.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding="same", activation="relu")
        self.batch_normalization_1 = K.layers.BatchNormalization()
        self.conv_2 = K.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu")
        self.batch_normalization_2 = K.layers.BatchNormalization()
        self.flatten = K.layers.Flatten()
        self.dense = K.layers.Dense(3136, activation = "relu")
        self.batch_normalization_3 = K.layers.BatchNormalization()

    @tf.function
    def call(self, inputs, training):
        x = self.conv_1(inputs, training=training)
        x = self.batch_normalization_1(x, training=training)
        x = self.conv_2(x, training=training)
        x = self.batch_normalization_2(x, training=training)
        x = self.flatten(x, training=training)
        x = self.dense(x, training=training)
        x = self.batch_normalization_3(x, training=training)
        return x
