from MLP import MLP
import numpy as np
from matplotlib import pyplot as plt



data = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])

# target labels for each logical connective
target_and = np.array([0, 0, 0, 1])
target_or= np.array([0, 1, 1, 1])
target_nor = np.array([1, 0, 0, 0])
target_nand = np.array([1, 1, 1, 0])
target_xor = np.array([0, 1, 1, 0])


def train_and_evaluate(mlp, epochs, targets):
    accuracies = []
    losses = []
    for epoch in range(epochs):
        average_loss = 0
        accuracy = 0
        for i, input in enumerate(data):
            output = mlp.forward_step(input)
            loss = (targets[i] - output) ** 2
            average_loss += loss
            mlp.backprop_step(targets[i])
            accuracy += abs(targets[i] - round(output))
        average_loss /= 4
        accuracy = 1 - (accuracy / 4)
        accuracies.append(accuracy)
        losses.append(average_loss)
    return accuracies, losses

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
    plt.plot(range(epochs), epoch_evaluations)
    plt.xlabel("Epochs")
    plt.ylabel(evaluation_metric)
    plt.show()

epochs = 1000
and_mlp = MLP()
or_mlp = MLP()
nand_mlp = MLP()
nor_mlp = MLP()
xor_mlp = MLP()


and_accuracies, and_losses = train_and_evaluate(and_mlp, epochs, target_and)
or_accuracies, or_losses = train_and_evaluate(or_mlp, epochs, target_or)
nand_accuracies, nand_losses = train_and_evaluate(nand_mlp, epochs, target_nand)
nor_accuracies, nor_losses = train_and_evaluate(nor_mlp, epochs, target_nor)
xor_accuracies, xor_losses = train_and_evaluate(xor_mlp, epochs, target_xor)

visualize("and", "loss", epochs, and_losses)
visualize("and", "accuracy", epochs, and_accuracies)
visualize("or", "loss", epochs, or_losses)
visualize("or", "accuracy", epochs, or_accuracies)
visualize("nand", "loss", epochs, nand_losses)
visualize("nand", "accuracy", epochs, nand_accuracies)
visualize("nor", "loss", epochs, nor_losses)
visualize("nor", "accuracy", epochs, nor_accuracies)
visualize("xor", "loss", epochs, xor_losses)
visualize("xor", "accuracy", epochs, xor_accuracies)
visualize("xor", "accuracy", epochs, xor_accuracies)


