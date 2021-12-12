'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from lstm_layer import LSTM_Layer

class LSTM_Model(K.Model):
    '''
    LSTM Model containing one LSTM layer.
    Sublassing from K.Model.

    Attributes:
        lstm: LSTM Layer
        output_layer: Dense layer with 1 unit using sigmoid activation
    '''

    def __init__(self):
        '''
        Initializes a LSTM Model.
        '''
        super(LSTM_Model, self).__init__()

        self.lstm = LSTM_Layer()
        self.output_layer = K.layers.Dense(1, activation = "sigmoid")

    @tf.function
    def call(self, x):
        '''
        Passes input data through Network Layers.

        Args:
            x: input tensor containg input at t

        Returns:
            Prediction float of network, between 0 and 1
        '''
        x = self.lstm(x, self.lstm.zero_states(64))
        y = self.output_layer(x)

        return y