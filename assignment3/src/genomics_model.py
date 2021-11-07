'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
from genomics_dense_layer import GenomicsDenseLayer

class GenomicsModel(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model

    Structure:  
            Hidden Layer 1: 256 neurons, sigmoid activation
            Hidden Layer 2: 256 neurons, sigmoid activation
            Output Layer:     10 neurons, softmax activation  

    ...

        Attributes
        ----------
        hidden_1 : GenomicsDenseLayer
        hidden_2 : GenomicsDenseLayer
        output_layer : GenomicsDenseLayer
    '''

    # initialize model with two hidden layers and one output layer
    def __init__(self):

        super(GenomicsModel, self).__init__()

        self.hidden_1 = GenomicsDenseLayer(256, activation = tf.math.sigmoid)
        self.hidden_2 = GenomicsDenseLayer(256, activation = tf.math.sigmoid)

        self.output_layer = GenomicsDenseLayer(10, activation = tf.nn.softmax)

    # forward step
    def call(self, inputs):

        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        x = self.output_layer(x)

        return x