#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single Perceptron
@authors: lmcdonald
"""

import numpy as np

class Perceptron():
    """
    Singlpe perceptron

    args: 
            input_units: number of weights coming in to Perceptron
            alpha: optional learning rate with defaul = 1
    """

    # constructor
    def __init__(self, input_units, alpha = 1):

        self.input_units = input_units
        self.alpha = alpha
        pass

    # calculates activation of perceptron 
    # using sigmoid activation function
    def forward_step(self, inputs):

        pass

    # updates the parameters
    # computes gradients for weights and bias from error term delta
    def update(self, delta):

        pass

    
