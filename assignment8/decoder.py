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
        self.dense = K.layers.Dense(49, activation = "sigmoid")
        self.reshape = K.layers.Reshape((7,7,1))
        self.trans_conv_1 = K.layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = "same", activation = "relu")
        self.trans_conv_2 = K.layers.Conv2DTranspose(filters = 16, kernel_size = 3, strides = 2, padding = "same", activation = "relu")
        self.out = K.layers.Conv2D(filters = 1, kernel_size = 3, strides = 1, padding = "same", activation = "sigmoid")

    @tf.function
    def call(self, inputs):
        x = self.dense(inputs)
        x = self.reshape(x)
        x = self.trans_conv_1(x)
        x = self.trans_conv_2(x)
        x = self.out(x)
        return x