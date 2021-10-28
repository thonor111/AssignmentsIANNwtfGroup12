#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Perceptron

@authors: lmcdonald
"""

import numpy as np
from activation_functions import sigmoid

class Perceptron():
    """
    A single perceptron.

    ...

        Attributes
        ----------
        input_units : int
            number of weights reaching the perceptron
        alpha : float
            learning rate
        weights : 1D array
            the weights of each input reaching the perceptron
            random initialization
        bias : float
            the bias
            random initialization
        weighted_input : float
            inputs * weights + bias
        output: float
            the perceptrons output
        inputs : numpy array
            the clean input values to the perceptron   
    """

    def __init__(self, input_units, alpha = 1.0):
        '''
        Initializes a Perceptron.

            Parameters :
                    input_units (int) :
                        number of weights reaching the perceptron
                    alpha (float) :
                        learning rate; default = 1
        '''

        self.input_units = input_units
        self.alpha = alpha
        self.weights = np.random.randn(self.input_units)
        self.bias = np.random.randn()
        self.weighted_input = 0.0
        self.output = 0.0
        self.inputs = np.array([])


    def forward_step(self, inputs):
        '''
        Calculates the activation of the perceptron using the sigmoid
        activation function.

            Parameters :
                    inputs (array) : Inputs to the perceptron

            Returns :
                    self.output (float): Output of the perceptron
        '''

        self.inputs = inputs
        self.weighted_input = self.weights @ inputs + self.bias
        self.output = sigmoid(self.weighted_input)

        return self.output


    def update(self, delta):
        '''
        Computes gradients for weights and bias from error term delta
        and updates the parameters accordingly.

            Parameters :
                    delta (float) : error term from previous layer
        '''

        delta_weights = [None]*self.input_units
        gradient = delta * self.alpha

        for i, weight in enumerate(self.weights):

            delta_weights[i] = gradient * self.inputs[i]
            self.weights[i] = weight - self.alpha * delta_weights[i]

        self.bias = self.bias - self.alpha * gradient
    
