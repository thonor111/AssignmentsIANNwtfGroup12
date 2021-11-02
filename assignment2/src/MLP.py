#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multilayer Perceptron

@authors: lmcdonald, hsanna
"""

import numpy as np
from Perceptron import Perceptron
from activation_functions import sigmoidprime

class MLP():
    """
    A multilayer perceptron.
    Structure: 
            2D Input 
            1 hidden layer with 4 neurons
            1 output neuron

    ...

        Attributes
        ----------
        hidden : array
            hidden layer with 4 perceptrons
        output_neuron : Perceptron
            output neuron
        output : float
            final output value  
    """

    def __init__(self):
        '''
        Initializes the MLP.
        '''

        self.hidden = np.array(
            [Perceptron(input_units = 2), Perceptron(input_units = 2), 
             Perceptron(input_units = 2), Perceptron(input_units = 2)])
        self.output_neuron = Perceptron(input_units = 4)
        self.output = 0.0


    def forward_step(self, inputs):
        '''
        Passes the inputs through the network.

            Parameters :
                    inputs (array) :
                        input data
            Returns : 
                    output (float) :
                        the output value after a single forward pass
        '''

        hidden_activations = []

        for perceptron in self.hidden:

            hidden_activations.append(perceptron.forward_step(inputs))
    
        hidden_activations = np.reshape(hidden_activations, newshape=(-1))

        self.output = self.output_neuron.forward_step(hidden_activations)

        return self.output


    def backprop_step(self, target):
        '''
        One backpropagation pass through the MLP, updating the weights and biases.

            Parameters :
                    target (int) :
                        target output
        '''

        delta_output_neuron = - (target - self.output) * sigmoidprime(self.output_neuron.weighted_input)
        self.output_neuron.update(delta_output_neuron)

        for i, perceptron in enumerate(self.hidden):

            delta_perceptron = self.output_neuron.weights[i] * delta_output_neuron * sigmoidprime(perceptron.weighted_input)
            perceptron.update(delta_perceptron)