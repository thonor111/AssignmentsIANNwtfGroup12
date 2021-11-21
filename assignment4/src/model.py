'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
from dense_layer import DenseLayer

class Model(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model

    Structure:  
            Hidden Layer 1: 256 neurons, sigmoid activation
            Hidden Layer 2: 256 neurons, sigmoid activation
            Output Layer:     10 neurons, softmax activation  

    ...

        Attributes
        ----------
        hidden_1 : DenseLayer
        hidden_2 : DenseLayer
        output_layer : DenseLayer
    '''

    # initialize model with two hidden layers and one output layer
    def __init__(self):
        '''
            Initializes hidden and output layers of the model
        '''

        super(Model, self).__init__()

        self.hidden_1 = tf.keras.layers.Dense(128, activation = tf.math.sigmoid, kernel_regularizer = "l2")
        self.hidden_2 = tf.keras.layers.Dense(128, activation = tf.math.sigmoid, kernel_regularizer = "l2")

        # binary output layer
        self.output_layer = tf.keras.layers.Dense(1, activation = tf.nn.sigmoid)

        self.dropout_layer = tf.keras.layers.Dropout(rate = 0.1)

    # forward step, calculate prediction
    def call(self, inputs, training):
        '''
            Forward Step
            Passes activations through the network and calculates prediction

            Args:
                inputs: the inputs to the model

            Returns:
                y: the prediction of the model
        '''

        x = self.hidden_1(inputs)
        x = self.dropout_layer(x, training = training)
        x = self.hidden_2(x)
        y = self.output_layer(x)

        return y