'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf

def load_data():
    """
    Loading and preprocessing the data.
        Returns:
          - train_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our training dataset
          - valid_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our validation dataset
          - test_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our test dataset
    """

    train_ds, valid_ds, test_ds, = tfds.load(name="cifar10", split=['train[0%:80%]','train[80%:100%]','test'], as_supervised=True)

    train_ds = prepare_datapipeline(train_ds)
    valid_ds = prepare_datapipeline(valid_ds)
    test_ds = prepare_datapipeline(test_ds)

    return train_ds, valid_ds, test_ds


def train_step(model, input, target, loss_function, optimizer):
    """
    Performs a forward and backward pass for  one dataponit of our training set
      Args:
        - model: <tensorflow.keras.Model> our created MLP model
        - input: <tensorflow.tensor> our input
        - target: <tensorflow.tensor> our target
        - loss_funcion: <keras function> function we used for calculating our loss
        - optimizer: <keras function> our optimizer used for backpropagation
      Returns:
        - loss: <float> our calculated loss for the datapoint
      """

    with tf.GradientTape() as tape:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # calculaing the gradients
        gradients = tape.gradient(loss, model.trainable_variables)

    # updating weights and biases
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss

def test(model, test_data, loss_function):
    """
    Test our MLP, by going through our testing dataset,
    performing a forward pass and calculating loss and accuracy
      Args:
        - model: <tensorflow.keras.Model> our created MLP model
        - test_data: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our preprocessed test dataset
        - loss_funcion: <keras function> function we used for calculating our loss
      Returns:
          - loss: <float> our mean loss for this epoch
          - accuracy: <float> our mean accuracy for this epoch
    """

    # initializing lists for accuracys and loss
    accuracy_aggregator = []
    loss_aggregator = []

    for (input, target) in test_data:

        # forward step
        prediction = model(input)

        # calculating loss
        loss = loss_function(target, prediction)

        # add loss and accuracy to the lists
        loss_aggregator.append(loss.numpy())
        for t, p in zip(target, prediction):
            accuracy_aggregator.append(
                tf.cast(tf.math.argmax(t) == tf.math.argmax(p), tf.float32))

    # calculate the mean of the loss and accuracy (for this epoch)
    loss = tf.reduce_mean(loss_aggregator)
    accuracy = tf.reduce_mean(accuracy_aggregator)

    return loss, accuracy

def visualize(train_losses, valid_losses, valid_accuracies):
    """
    Displays the losses and accuracies from the different models in a plot-grid.
      Args:
        - train_losses: <list> mean training losses per epoch
        - valid_losses: <list> mean testing losses per epoch
        - valid_accuracies: <list> mean accuracies (testing dataset) per epoch
    """
    plt.figure()
    line1, = plt.plot(train_losses)
    line2, = plt.plot(valid_losses)
    line3, = plt.plot(valid_accuracies)
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.ylim(top = 1)
    plt.ylim(bottom = 0)
    plt.legend((line1,line2, line3),(" Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"))
    plt.xlabel("Training epoch")
    plt.show()

def visualize_2(train_losses, valid_losses, valid_accuracies):
    fig, axs = plt.subplots(2,1)

    axs[0].plot(train_losses)
    axs[0].plot(valid_losses)
    axs[1].plot(valid_accuracies)

    fig.legend([" Train_ds loss", " Valid_ds loss", " Valid_ds accuracy"])
    plt.xlabel("Training epoch")
    fig.tight_layout()
    plt.show()
 


def classify(model, optimizer, num_epochs, train_ds, valid_ds):
    """
    Trains and tests our predefined model.
        Args:
            - model: <tensorflow.keras.Model> our untrained model
            - optimizer: <keras function> optimizer for the model
            - num_epochs: <int> number of training epochs
            - train_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our training dataset
            - valid_ds: <tensorflow.python.data.ops.dataset_ops.PrefetchDataset> our validation set for testing and optimizing hyperparameters
        Returns:
            - results: <list<list<float>>> list with losses and accuracies
            - model: <tensorflow.keras.Model> our trained MLP model
    """

    tf.keras.backend.clear_session()

    # initialize the loss: categorical cross entropy
    cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

    # initialize lists for later visualization.
    train_losses = []
    valid_losses = []
    valid_accuracies = []

    # testing on our valid_ds once before we begin
    valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

    # Testing on our train_ds once before we begin
    train_loss, _ = test(model, train_ds, cross_entropy_loss)
    train_losses.append(train_loss)

    # training our model for num_epochs epochs.
    for epoch in range(num_epochs):
        print(
            f'Epoch: {str(epoch+1)} starting with (validation set) accuracy {valid_accuracies[-1]} and loss {valid_losses[-1]}')

        # training (and calculating loss while training)
        epoch_loss_agg = []

        for input, target in train_ds:
            train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
            epoch_loss_agg.append(train_loss)

        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        print(f'Epoch: {str(epoch+1)} train loss: {train_losses[-1]}')

        # testing our model in each epoch to track accuracy and loss on the validation set
        valid_loss, valid_accuracy = test(model, valid_ds, cross_entropy_loss)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)

    results = [train_losses, valid_losses, valid_accuracies]
    return results, model