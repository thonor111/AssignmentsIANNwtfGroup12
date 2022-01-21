'''
authors: tnortmann, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K


class Generator(K.Model):

    def __init__(self):
        '''
        Initializes the generator
        '''
        super(Generator, self).__init__()
        self.dense = K.layers.Dense(49, activation="sigmoid")
        self.reshape = K.layers.Reshape((7,7,1))
        self.trans_conv_1 = K.layers.Conv2DTranspose(filters = 64, kernel_size = 4, strides = 1, padding = "same", activation = "relu")
        self.batch_normalization_1 = K.layers.BatchNormalization()
        self.trans_conv_2 = K.layers.Conv2DTranspose(filters=64, kernel_size=4, strides=2, padding="same",activation="relu")
        self.batch_normalization_2 = K.layers.BatchNormalization()
        self.trans_conv_3 = K.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=1, padding="same",activation="relu")
        self.batch_normalization_3 = K.layers.BatchNormalization()
        self.trans_conv_4 = K.layers.Conv2DTranspose(filters=32, kernel_size=4, strides=2, padding="same",activation="relu")
        self.batch_normalization_4 = K.layers.BatchNormalization()
        self.trans_conv_5 = K.layers.Conv2DTranspose(filters=16, kernel_size=4, strides=1, padding="same",activation="relu")
        self.batch_normalization_5 = K.layers.BatchNormalization()
        self.trans_conv_6 = K.layers.Conv2DTranspose(filters=1, kernel_size=4, strides=1, padding="same",activation="sigmoid")


    @tf.function
    def call(self, inputs, training):
        x = self.dense(inputs, training=training)
        x = self.reshape(x, training=training)
        x = self.trans_conv_1(x, training=training)
        x = self.batch_normalization_1(x, training=training)
        x = self.trans_conv_2(x, training=training)
        x = self.batch_normalization_2(x, training=training)
        x = self.trans_conv_3(x, training=training)
        x = self.batch_normalization_3(x, training=training)
        x = self.trans_conv_4(x, training=training)
        x = self.batch_normalization_4(x, training=training)
        x = self.trans_conv_5(x, training=training)
        x = self.batch_normalization_5(x, training=training)
        x = self.trans_conv_6(x, training=training)
        return x