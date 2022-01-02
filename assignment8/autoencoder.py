'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from decoder import Decoder
from encoder import Encoder

class Autoencoder(K.Model):

    def __init__(self):
        '''
        Initializes the autoencoder
        '''
        super(Autoencoder, self).__init__()
        self.encoder = Encoder()
        self.bottleneck = K.layers.Dense(10, activation = "sigmoid")
        self.decoder = Decoder()

    @tf.function
    def call(self, x):
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x