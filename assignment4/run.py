import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import pandas as pd
import datapipeline
import Models
import training_testing
import visualisation

#Readind the data in a pandas dataframe with ; as a separator
wine_quality= pd.read_csv("winequality-red.csv", ";")

#Datapipeline: splitting the dataset, target for a binary classification
train_input, test_input, validate_input = datapipeline.data_pipeline(wine_quality)

### Hyperparameters
num_epochs = 10
learning_rate = 0.01

# Initialize the model.
model = Models.MyModel()
#initialize the model with drop out
model_drop = Models.MyModel_dropout()

# Initialize the loss: binary cross entropy. (Binary classification: good or bad wine)
binary_entropy_loss = tf.keras.losses.BinaryCrossentropy()


# Initialize the optimizer: SGD with default parameters.
optimizer = tf.keras.optimizers.SGD(learning_rate)
# Initialize the optimizer: Adam
opt = tf.keras.optimizers.Adam(learning_rate= 0.1)
# Initialize the optimizer: SGD with momentum
opt_1 = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.5)




# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []


# Initialize lists for Adam optimezer.
train_losses_adam = []
validate_losses_adam = []
validate_accuracies_adam = []

# Initialize lists for SGD with momentum.
train_losses_momentum = []
validate_losses_momentum = []
validate_accuracies_momentum = []



# Initialize lists for the model with dropout.
train_losses_dropout = []
validate_losses_dropout = []
validate_accuracies_dropout = []

#testing once before we begin
test_loss, test_accuracy = training_testing.test(model, test_input, binary_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = training_testing.test(model, train_input, binary_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_input:
        train_loss = training_testing.train_step(model, input, target, binary_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = training_testing.test(model, test_input, binary_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)


# Training loop with Adam optimer
for epoch in range(num_epochs):
    if epoch >= 1:
        print(f'Epoch: {str(epoch)} starting with accuracy {validate_accuracies_adam[-1]}')
    else:
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg_a = []
    for input,target in train_input:
        train_loss = training_testing.train_step(model, input, target, binary_entropy_loss, opt)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses_adam.append(tf.reduce_mean(epoch_loss_agg_a))

    #validating, so we can track accuracy and test loss
    validate_loss, validate_accuracy = training_testing.test(model, validate_input, binary_entropy_loss)
    validate_losses_adam.append(validate_loss)
    validate_accuracies_adam.append(validate_accuracy)

#Training loop with SGD with momentum
for epoch in range(num_epochs):
    if epoch >= 1:
        print(f'Epoch: {str(epoch)} starting with accuracy {validate_accuracies_momentum[-1]}')
    else:
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg_m = []
    for input,target in train_input:
        train_loss = training_testing.train_step(model, input, target, binary_entropy_loss, opt_1)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses_momentum.append(tf.reduce_mean(epoch_loss_agg_m))

    #testing, so we can track accuracy and test loss
    validate_loss, validate_accuracy = training_testing.test(model, validate_input, binary_entropy_loss)
    validate_losses_momentum.append(validate_loss)
    validate_accuracies_momentum.append(validate_accuracy)



#Training loop using the model with drop out
for epoch in range(num_epochs):
    if epoch >= 1:
        print(f'Epoch: {str(epoch)} starting with accuracy {validate_accuracies_dropout[-1]}')
    else:
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg_1 = []
    for input,target in train_input:
        train_loss = training_testing.train_step(model_drop, input, target, binary_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses_dropout.append(tf.reduce_mean(epoch_loss_agg_1))

    #validating, so we can track accuracy and validate loss
    validate_loss, validate_accuracy = training_testing.test(model_drop, validate_input, binary_entropy_loss)
    validate_losses_dropout.append(validate_loss)
    validate_accuracies_dropout.append(validate_accuracy)


visualisation.visualize(train_losses, test_losses, test_accuracies)
visualisation.visualize(train_losses_adam, validate_losses_adam, validate_accuracies_adam)
visualisation.visualize(train_losses_momentum, validate_losses_momentum, validate_accuracies_momentum)
visualisation.visualize(train_losses_dropout, validate_losses_dropout, validate_accuracies_dropout)


