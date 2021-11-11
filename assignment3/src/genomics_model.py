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
        '''
            Initializes hidden and output layers of the model
        '''

        super(GenomicsModel, self).__init__()

        self.hidden_1 = GenomicsDenseLayer(256, activation = tf.math.sigmoid)
        self.hidden_2 = GenomicsDenseLayer(256, activation = tf.math.sigmoid)

        self.output_layer = GenomicsDenseLayer(10, activation = tf.nn.softmax)

    # forward step, calculate prediction
    def call(self, inputs):
        '''
            Forward Step
            Passes activations through the network and calculates prediction

            Args:
                inputs: the inputs to the model

            Returns:
                y: the prediction of the model
        '''

        x = self.hidden_1(inputs)
        x = self.hidden_2(x)
        y = self.output_layer(x)

        return y