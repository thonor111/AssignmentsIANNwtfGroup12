'''
authors: tnortmann, hsanna, lmcdonald
'''

from generator import my_integration_task
import tensorflow as tf
import tensorflow.keras as K
from lstm_model import LSTM_Model
import matplotlib.pyplot as plt
import input_pipeline
import training_loop

# generate dataset
dataset = tf.data.Dataset.from_generator(
    my_integration_task, 
    output_signature=(
        tf.TensorSpec(shape=(25, 1), dtype=tf.float32), 
        tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )

num_samples = 32000

num_train_samples = int(0.8 * num_samples)
num_valid_samples = int(0.1 * num_samples)
num_test_samples = int(0.1 * num_samples)

# split into train, valid and test
train_data = dataset.take(num_train_samples)
valid_data = dataset.skip(num_train_samples).take(num_valid_samples)
test_data = dataset.skip(num_train_samples).skip(
    num_valid_samples).take(num_test_samples)

# prepare data
train_data = train_data.apply(input_pipeline.prepare_data)
valid_data = valid_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1

# Initialize Model
model = LSTM_Model()

# loss function
loss_function = K.losses.MeanSquaredError()

# optimizer
optimizer = K.optimizers.SGD(alpha)

# initialize lists for later visualization.
train_losses = []
valid_losses = []
valid_accuracies = []

# testing once before we begin
valid_loss, valid_accuracy = training_loop.test(
    model, valid_data, loss_function)
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
line1 = plt.plot(valid_accuracies)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(top=1)
plt.ylim(bottom=0)
plt.legend([line1], ["Validation accuracy"])
plt.show()
