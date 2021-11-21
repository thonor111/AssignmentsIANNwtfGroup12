'''
authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf
import tensorflow_datasets as tfds
import input_pipeline, training_loop
from model import Model
import matplotlib.pyplot as plt
import pandas as pd

tf.keras.backend.clear_session()

# load dataset 
dataset = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", delimiter = ";")

# normalize
max_vals = dataset.max()
min_vals = dataset.min()
dataset.iloc[:,:-2] = ( dataset.iloc[:,:-2] - dataset.iloc[:,:-2].min() ) / ( dataset.iloc[:,:-2].max() - dataset.iloc[:,:-2].min() )

# OPTIONAL
# check to take a look at the shape of the dataset
#print(dataset.info())
#print(dataset.head())

# split dataset into train, validation and test sets
train_data = dataset.sample(frac = 0.6, random_state = 1)
validation_data = dataset.drop(train_data.index).sample(frac = 0.5, random_state = 1)
test_data = dataset.drop(train_data.index).drop(validation_data.index)

# OPTIONAL
# validate splits of dataset
#print(dataset.shape, train_data.shape, validation_data.shape, test_data.shape)

# seperate labels from input
train_input = train_data.drop(columns=['quality'], axis=1)
train_labels = train_data.drop(train_input.columns, axis=1)
validation_input = validation_data.drop(columns=['quality'], axis=1)
validation_labels = validation_data.drop(validation_input.columns, axis=1)
test_input = test_data.drop(columns=['quality'], axis=1)
test_labels = test_data.drop(test_input.columns, axis=1)
#print(labels.head())

train_ds = tf.data.Dataset.from_tensor_slices((train_input, train_labels))
train_ds_np = tfds.as_numpy(train_ds)
validation_ds = tf.data.Dataset.from_tensor_slices((validation_input, validation_labels))
validation_ds_np = tfds.as_numpy(validation_ds)
test_ds = tf.data.Dataset.from_tensor_slices((test_input, test_labels))
test_ds_np = tfds.as_numpy(test_ds)
# for data in test_ds_np:
#     print(data)

# apply input pipeline to train and test dataset splits
train_ds = train_ds.apply(input_pipeline.prepare_data)
validation_ds = validation_ds.apply(input_pipeline.prepare_data)
test_ds = test_ds.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1

# Initialize Model
model = Model()

# loss function
loss_function = tf.keras.losses.BinaryCrossentropy()

# vanilla SGD
optimizer = tf.keras.optimizers.SGD(learning_rate = alpha)
# SGD with momentum
#optimizer = tf.keras.optimizers.SGD(learning_rate = alpha, momentum = 0.2)
# SGD with Nesterov
#optimizer = tf.keras.optimizers.SGD(learning_rate = alpha, nesterov = True)
# SGD with Nesterov and momentum
#optimizer = tf.keras.optimizers.SGD(learning_rate = alpha, momentum = 0.2, nesterov = True)

# ADAM optimizer
#optimizer = tf.keras.optimizers.Adam(learning_rate = alpha)

# Initialize lists for later visualization.
train_losses = []
test_losses = []
test_accuracies = []
validation_losses = []
validation_accuracies = []

# testing once on the validation set before we begin
validation_loss, validation_accuracy = training_loop.test(model, validation_ds, loss_function)
validation_losses.append(validation_loss)
validation_accuracies.append(validation_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = training_loop.test(model, train_ds, loss_function)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # print out starting accuracy
    print(f'Epoch: {str(epoch)} starting with accuracy {validation_accuracies[-1]}')

    # training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_ds:
        train_loss = training_loop.train_step(model, input, target, loss_function, optimizer)
        epoch_loss_agg.append(train_loss)
    
    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing, so we can track accuracy and test loss
    validation_loss, validation_accuracy = training_loop.test(model, validation_ds, loss_function)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(validation_losses)
line3, = plt.plot(validation_accuracies)
plt.xlabel("Training steps")
plt.ylabel("Loss/Accuracy")
plt.legend((line1,line2, line3),("training loss","validation loss", "validation accuracy"))
plt.show()