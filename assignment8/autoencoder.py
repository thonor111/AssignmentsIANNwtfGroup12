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
        self.dense1 = K.layers.Dense(500, activation = "sigmoid")
        self.dense2 = K.layers.Dense(100, activation="sigmoid")
        self.bottleneck = K.layers.Dense(embedding_dimensions, activation = "sigmoid")
        self.dense3 = K.layers.Dense(100, activation="sigmoid")
        self.dense4 = K.layers.Dense(500, activation="sigmoid")
        self.decoder = Decoder()

    @tf.function
    def call(self, x):
        x = self.encoder(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.bottleneck(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.decoder(x)
        return x

    @tf.function
    def encode(self, x):
        x = self.encoder(x)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.bottleneck(x)
        return x