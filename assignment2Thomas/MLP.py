import numpy as np
from Perceptron import Perceptron

class MLP:

    def __init__(self):
        self.hidden_layer = np.array([Perceptron(2), Perceptron(2), Perceptron(2), Perceptron(2)])
        self.output_neuron = Perceptron(4)


    def forward_step(self, input):
        outputs_hidden_layer = np.array([])
        for i in range(4):
            outputs_hidden_layer = np.append(outputs_hidden_layer, self.hidden_layer[i].forward_step(input))
        return self.output_neuron.forward_step(np.array(outputs_hidden_layer))

    def backprop_step(self, target):
        delta_output_neuron = (self.output_neuron.output - target) * self.output_neuron.output * (1 - self.output_neuron.output)
        self.output_neuron.update(delta_output_neuron)
        for i, perceptron in enumerate(self.hidden_layer):
            delta = self.output_neuron.weights[i] * delta_output_neuron * perceptron.output * (1 - perceptron.output)
            perceptron.update(delta)