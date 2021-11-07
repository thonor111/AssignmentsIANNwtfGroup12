'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow_datasets as tfds
import input_pipeline, training_loop
from genomics_model import GenomicsModel
import matplotlib.pyplot as plt

tf.keras.backend.clear_session()

# load dataset and split into to train (100,000 examples) and test (1,000 examples) sets
train_data, test_data = tfds.load('genomics_ood', split=['train[:10%]', 'test[:1%]'], as_supervised=True)

# apply input pipeline to train and test dataset splits
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

train_dataset_numpy = tfds.as_numpy(train_data)

for i in train_dataset_numpy:
  print(i)
  break

# Hyperparameters
num_epochs = 10
alpha = 0.1

# Initialize Model
model = GenomicsModel()

# loss function
categorical_cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()

# optimizer
optimizer = tf.keras.optimizers.SGD(alpha)

# Initialize lists for later visualization.
train_losses = []

test_losses = []
test_accuracies = []

#testing once before we begin
test_loss, test_accuracy = training_loop.test(model, test_data, categorical_cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

#check how model performs on train data once before we begin
train_loss, _ = training_loop.test(model, test_data, categorical_cross_entropy_loss)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    #training (and checking in with training)
    epoch_loss_agg = []
    for input,target in train_data:
        train_loss = training_loop.train_step(model, input, target, categorical_cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)
    
    #track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    #testing, so we can track accuracy and test loss
    test_loss, test_accuracy = training_loop.test(model, test_data, categorical_cross_entropy_loss)
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