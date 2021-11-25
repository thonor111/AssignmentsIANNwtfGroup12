import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import data_pipeline
from model import MyModel
from training_testing import train_step, test
import visualisation

#Importing the fashon mnist dataset
train_ds, test_ds = tfds.load('fashion_mnist', split=['train', 'test'], as_supervised=True)

#applying the data pipeline
train_dataset = train_ds.apply(data_pipeline.prepare_mnist_data)
test_dataset = test_ds.apply(data_pipeline.prepare_mnist_data)

### Hyperparameters
num_epochs = 10
learning_rate = 0.1

# Initialize the model.
model= MyModel()
# Initialize the loss: categorical cross entropy.
categorical_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
# Initialize the optimizer: SGD with default parameters.
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = test(model, test_dataset, categorical_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = test(model, train_dataset, categorical_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_dataset:
        train_loss = train_step(model, input, target, categorical_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = test(model, test_dataset, categorical_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

    visualisation.visualize(train_losses, test_losses, test_accuracies)

