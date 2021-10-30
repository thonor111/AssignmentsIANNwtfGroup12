import numpy as np

class Perceptron:

    def __init__(self, input_units):
        self.input_units = input_units
        self.weights = np.random.rand(input_units)
        self.bias = np.random.random()
        self.alpha = 1
        self.input = 0
        self.output = 0

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def forward_step(self, inputs):
        self.input = inputs
        weighted_input = float((self.weights.T @ np.matrix(inputs).T) + self.bias)
        self.output = self.sigmoid(weighted_input)
        return self.output

    def update(self, delta):
        gradients = self.input * delta
        self.weights = self.weights - (self.alpha * gradients)
        self.bias = self.bias - self.alpha * delta
