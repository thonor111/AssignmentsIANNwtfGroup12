'''
authors: tnortmann, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K


class Discriminator(K.Model):

    def __init__(self):
        '''
        Initializes the discriminator
        '''
        super(Discriminator, self).__init__()
        self.conv_1 = K.layers.Conv2DTranspose(filters=16, kernel_size=3, strides=1, padding="same", activation="relu")
        self.batch_normalization_1 = K.layers.BatchNormalization()
        self.pooling_1 = K.layers.AveragePooling2D(pool_size = 2)
        self.conv_2 = K.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding="same", activation="relu")
        self.batch_normalization_2 = K.layers.BatchNormalization()
        self.pooling_2 = K.layers.AveragePooling2D(pool_size=2)
        self.conv_3 = K.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=1, padding="same", activation="relu")
        self.batch_normalization_3 = K.layers.BatchNormalization()
        self.pooling_3 = K.layers.AveragePooling2D(pool_size=2)
        self.global_pooling = K.layers.GlobalMaxPooling2D()
        self.flatten = K.layers.Flatten()
        self.out = K.layers.Dense(1, activation="sigmoid")

    @tf.function
    def call(self, inputs, training):
        x = self.conv_1(inputs, training = training)
        x = self.batch_normalization_1(x, training = training)
        x = self.pooling_1(x, training = training)
        x = self.conv_2(x, training=training)
        x = self.batch_normalization_2(x, training=training)
        x = self.pooling_2(x, training=training)
        x = self.conv_3(x, training=training)
        x = self.batch_normalization_3(x, training=training)
        x = self.pooling_3(x, training=training)
        x = self.global_pooling(x, training = training)
        x = self.flatten(x, training = training)
        x = self.out(x, training = training)
        return x