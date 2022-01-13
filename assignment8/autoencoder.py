'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from decoder import Decoder
from encoder import Encoder

class Autoencoder(K.Model):

    def __init__(self, embedding_dimensions = 10):
        '''
        Initializes the autoencoder
        '''
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = K.layers.Dense(embedding_dimensions, activation = "sigmoid")
        self.decoder = Decoder()

    @tf.function
    def call(self, x, training):
        x = self.encoder(x, training)
        x = self.bottleneck(x, training=training)
        x = self.decoder(x, training)
        return x

    @tf.function
    def encode(self, x, training = False):
        x = self.encoder(x, training)
        x = self.bottleneck(x, training=training)
        return x

    @tf.function
    def decode(self, x, training = False):
        x = self.decoder(x, training = training)
        return x