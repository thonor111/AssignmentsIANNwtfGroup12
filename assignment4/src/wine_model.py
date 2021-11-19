'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
from wine_dense_layer import WineDenseLayer
import numpy as np

class WineModel(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model

    Structure:  
            Hidden Layer 1: 256 neurons, sigmoid activation
            Hidden Layer 2: 256 neurons, sigmoid activation
            Output Layer:     10 neurons, softmax activation  

    ...

        Attributes
        ----------
        hidden_1 : WineDenseLayer
        hidden_2 : WineDenseLayer
        output_layer : WineDenseLayer
    '''

    # initialize model with two hidden layers and one output layer
    def __init__(self, p_gaussian_dropout):
        '''
            Initializes hidden and output layers of the model
        '''

        super(WineModel, self).__init__()

        self.hidden_1 = WineDenseLayer(80, activation_function = tf.keras.activations.sigmoid)
        self.hidden_2 = WineDenseLayer(80, activation_function = tf.keras.activations.sigmoid)

        self.output_layer = WineDenseLayer(1, activation_function = tf.keras.activations.sigmoid)

        self.std_layers = np.sqrt((1 - p_gaussian_dropout) / p_gaussian_dropout)

    # forward step, calculate prediction
    def call(self, inputs, dropout = True):
        '''
            Forward Step
            Passes activations through the network and calculates prediction

            Args:
                inputs: the inputs to the model

            Returns:
                y: the prediction of the model
        '''

        x = self.hidden_1(inputs, self.std_layers, dropout = dropout)
        x = self.hidden_2(x, self.std_layers, dropout = dropout)
        y = self.output_layer(x, self.std_layers, dropout = dropout)

        return y