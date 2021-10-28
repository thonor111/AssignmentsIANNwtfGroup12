#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors: lmcdonald
"""

import numpy as np
from numpy.core.fromnumeric import mean
from MLP import MLP
from matplotlib import pyplot as plt

def train_on_connective(mlp, num_epochs, target_values):
    '''
        Trains the MLP for the given connective

            Parameters :
                    mlp (MLP) :
                        the multilayer perceptron to be trained
                    num_epochs (int) :
                        the number of epochs to train the MLP for
                    target_values (array) :
                        the target outputs for a given connective given an input
            Returns : 
                    epoch_accuarcy (array) :
                        average accuracies for each training epoch
                    epoch_squared_error_loss (array) :
                        average squared error losses for each training epoch
        '''

    epochs = num_epochs
    epoch_accuracy = []
    epoch_squared_error_loss = []

    for epoch in range(epochs):

        avg_accuracy = 0.0
        avg_squared_error_loss = 0.0

        for i, datapoint in enumerate(data):

            output = mlp.forward_step(datapoint)
            mlp.backprop_step(target_and[i])

            avg_squared_error_loss += (target_values[i] - output)**2
            
            if (output >= 0.5 and target_values[i] == 1) or (output < 0.5 and target_values[i] == 0):
                avg_accuracy += 1

        epoch_accuracy.append(avg_accuracy/data.shape[0])
        epoch_squared_error_loss.append(avg_squared_error_loss/data.shape[0])

    return epoch_accuracy, epoch_squared_error_loss


def visualize(logical_connective, evaluation_metric, epochs, epoch_evaluations):
    '''
        Visualizes an evaluation metric against the epochs for a given connective

            Parameters :
                    logical_connective (string) :
                        the label of the plot; connective of choice
                    evaluation_metric (string) :
                        the label of the y-axis; chosen metric to visualize
                    epochs (array) :
                        array of ascending number up to max number of epochs to visualize
                    epoch_evaluations (array) :
                        array containg metric results for each epoch
        '''

    plt.figure()
    plt.title(logical_connective)
    plt.plot(epochs, epoch_evaluations)
    plt.xlabel("Epochs")
    plt.ylabel(evaluation_metric)
    plt.show()

# dataset with binary combinations
data = np.array(([0, 0],[0, 1],[1, 0],[1, 1]))

# target labels for each logical connective
target_and = np.array([0, 0, 0, 1])
target_or= np.array([0, 1, 1, 1])
target_nor = np.array([1, 0, 0, 0])
target_nand = np.array([1, 1, 1, 0])
target_xor = np.array([0, 1, 1, 0])

# initialize MLP for each logical connective
and_mlp = MLP()
or_mlp = MLP()
nor_mlp = MLP()
nand_mlp = MLP()
xor_mlp = MLP()

# train the MLPs on their respective connective
and_accuracies, and_losses = train_on_connective(and_mlp, 1000, target_and)
or_accuracies, or_losses = train_on_connective(or_mlp, 1000, target_or)
nor_accuracies, nor_losses = train_on_connective(nor_mlp, 1000, target_nor)
nand_accuracies, nand_losses = train_on_connective(nand_mlp, 1000, target_nand)
nor_accuracies, nor_losses = train_on_connective(xor_mlp, 1000, target_xor)

# visualize 
visualize("and", "loss", np.arange(len(and_losses)), and_losses)
visualize("and", "loss", np.arange(len(and_losses)), and_losses)

visualize("and", "loss", np.arange(len(and_losses)), and_losses)
visualize("and", "loss", np.arange(len(and_losses)), and_losses)

visualize("and", "loss", np.arange(len(and_losses)), and_losses)
visualize("and", "loss", np.arange(len(and_losses)), and_losses)

visualize("and", "loss", np.arange(len(and_losses)), and_losses)
visualize("and", "loss", np.arange(len(and_losses)), and_losses)

visualize("and", "loss", np.arange(len(and_losses)), and_losses)
visualize("and", "loss", np.arange(len(and_losses)), and_losses)



