'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import input_pipeline, training_loop
from wine_model import WineModel
import matplotlib.pyplot as plt
import pandas as pd

tf.keras.backend.clear_session()

# load dataset and split into to train (1400 examples) and test (199 examples) sets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep = ';')
train_data, test_data = input_pipeline.create_tf_dataset(data)


#train_data = input_pipeline.prepare_data(train_data)
#test_data = input_pipeline.prepare_data(test_data)


# apply input pipeline to train and test dataset splits
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1
p_gaussian_dropout = 0.5

# Initialize Model
model = WineModel(p_gaussian_dropout)

# loss function
mean_squared_error = tf.keras.losses.MeanSquaredError()

# stochastic gradient descent optimizer
#sgd_optimizer = tf.keras.optimizers.Adam(alpha, beta_1 = 0.8, beta_2 = 0.99 )
sgd_optimizer = tf.keras.optimizers.Adagrad(alpha)

# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []

# testing once before we begin
test_loss, test_accuracy = training_loop.test(model, test_data, mean_squared_error)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = training_loop.test(model, test_data, mean_squared_error)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # print out starting accuracy
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_data:
        train_loss = training_loop.train_step(model, input, target, mean_squared_error, sgd_optimizer)
        epoch_loss_agg.append(train_loss)
    
    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    test_loss, test_accuracy = training_loop.test(model, test_data, mean_squared_error)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)


# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
line3, = plt.plot(test_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()