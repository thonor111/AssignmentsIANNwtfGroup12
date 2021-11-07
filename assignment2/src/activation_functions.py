#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activation Functions and their primes

@authors: lmcdonald, hsanna, tnortmann
"""

import math

def sigmoid(x):
    '''
        Sigmoid activation function.
        Formula: 𝜎(𝑥)=1/(1+𝑒^(−𝑥))

            Args : 
                x (float) : input value

            Returns :
                sigmoid(x) 
    '''

    return 1 / (1 + math.exp(-x))


def sigmoidprime(x):
    '''
        Derivative of sigmoid activation function.
        Formula: 𝜎(𝑥)⋅(1−𝜎(𝑥))

            Args : 
                x (float) : input value

            Returns :
                sigmoid'(x)
    '''

    return sigmoid(x) * (1 - sigmoid(x))