'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K


class Decoder(K.Model):

    def __init__(self):
        '''
        Initializes the decoder
        '''
        super(Decoder, self).__init__()
        self.dense = K.layers.Dense(3136, activation = "relu")
        self.batch_normalization_1 = K.layers.BatchNormalization()
        self.reshape = K.layers.Reshape((7,7,64))
        self.trans_conv_1 = K.layers.Conv2DTranspose(filters = 64, kernel_size = 3, strides = 2, padding = "same", activation = "relu")
        self.batch_normalization_2 = K.layers.BatchNormalization()
        self.trans_conv_2 = K.layers.Conv2DTranspose(filters = 32, kernel_size = 3, strides = 2, padding = "same", activation = "relu")
        self.batch_normalization_3 = K.layers.BatchNormalization()
        self.out = K.layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same", activation = "sigmoid")

    @tf.function
    def call(self, inputs, training):
        x = self.dense(inputs, training=training)
        x = self.batch_normalization_1(x, training=training)
        x = self.reshape(x, training=training)
        x = self.trans_conv_1(x, training=training)
        x = self.batch_normalization_2(x, training=training)
        x = self.trans_conv_2(x, training=training)
        x = self.batch_normalization_3(x, training=training)
        x = self.out(x, training=training)
        return x