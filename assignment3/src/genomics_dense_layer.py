'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

class GenomicsDenseLayer(tf.keras.layers.Layer):
    '''
        Custom Dense Layer subclassing from tf.keras.layers.Layer

        Attributes
        ----------
        units : int
            amount of neurons in the layer
        activation_function : callable
            the activation function used in the layer
    '''


    def __init__(self, units = 10, activation_function = tf.math.sigmoid):
        '''
            Initializes a GenomicsDenseLayer

            Args:
                units: the number of units the layer should have
                activation_function: the activation function to be used in the layer
        '''

        super(GenomicsDenseLayer, self).__init__()

        self.units = units
        self.activation_function = activation_function


    def build(self, input_shape):
        '''
            Initializes weights and biases for the layer

            Args:
                input_shape: shape of the input, determines number of weights and biases
        '''

        # initialize weight at each unit for each input
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)

        # initialize one bias for each unit
        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    def call(self, inputs): 
        '''
        Forward Step
        Calculates the layer's activation

        Args:
            inputs: the inputs to the layer

        Returns:
            x: the layer's activation
    '''

        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation_function(x)

        return x