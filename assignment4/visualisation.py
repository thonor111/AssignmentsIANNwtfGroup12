import matplotlib.pyplot as plt


def visualize(train_losses, test_losses, test_accuracies):
    '''
    Visualize accuracy and loss for training and test data.

    Args:
    train_losses: a list of rank 0 tensors
    test_losses: a list of rank 0 tensors
    test_accuracies: a list of rank 0 tensors
    '''
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(test_losses)
    line3, = plt.plot(test_accuracies)
    plt.xlabel("Training steps")
    plt.ylabel("Loss/Accuracy")
    plt.legend((line1,line2, line3),("training","test", "test accuracy"))
    return plt.show()