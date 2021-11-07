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
        activation : callable
            the activation function used in the layer
    '''

    # initialize layer
    def __init__(self, units = 10, activation = tf.math.sigmoid):

        super(GenomicsDenseLayer, self).__init__()

        self.units = units
        self.activation = activation

    # initialize weights and bias for the layer
    def build(self, input_shape):

        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                               initializer='random_normal',
                               trainable=True)

        self.b = self.add_weight(shape=(self.units,),
                               initializer='random_normal',
                               trainable=True)

    # calculate output of the neuron
    def call(self, inputs): 

        x = tf.matmul(inputs, self.w) + self.b
        x = self.activation(x)

        return x