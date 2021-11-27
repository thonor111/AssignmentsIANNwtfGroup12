import tensorflow as tf
import tensorflow_datasets as tfds
import pipeline, training_loop
from model import Model
import matplotlib.pyplot as plt
import tensorflow.keras as K

tf.keras.backend.clear_session()

# load and split dataset into train, validation and test split
train_data, valid_data, test_data = tfds.load('fashion_mnist', split =
                ['train[:83%]', 'train[83%:]', 'test'], as_supervised = True)

# apply input pipeline to dataset splits
train_data = train_data.apply(pipeline.prepare_data)
valid_data = valid_data.apply(pipeline.prepare_data)
test_data = test_data.apply(pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1

# Initialize Model
model = Model()

# loss function
loss_function = K.losses.CategoricalCrossentropy()

# optimizer
optimizer = K.optimizers.SGD(alpha)

# initialize lists for later visualization.
train_losses = []
valid_losses = []
valid_accuracies = []

# testing once before we begin
valid_loss, valid_accuracy = training_loop.test(model, test_data, loss_function)
valid_losses.append(valid_loss)
valid_accuracies.append(valid_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = training_loop.test(model, train_data, loss_function)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # print out starting accuracy
    print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies[-1]}')

    # training (and checking in with training)
    epoch_losses = []
    for input, target in train_data:
        train_loss = training_loop.train_step(model, input, target,
                                        loss_function, optimizer)
        epoch_losses.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))

    # testing, so we can track accuracy and test loss
    valid_loss, valid_accuracy = training_loop.test(model, valid_data,
                                                loss_function)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(valid_losses)
line3, = plt.plot(valid_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.ylim(top = 1)
plt.ylim(bottom = 0)
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()