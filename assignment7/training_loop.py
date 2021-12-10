'''
authors: tnortmann, hsanna, lmcdonald
'''

import numpy as np
import tensorflow as tf

def train_step(model, input, target, loss_function, optimizer):
  '''
    Performs the training step

    Args:
      model: the model to be trained
      input: the input data
      target: the targets corresponding to the input data
      loss_function: the loss_function to be used
      optimizer: the optimizer to be used

    Returns:
      loss: the loss of the current epoch
  '''

  # loss_object and optimizer_object are instances of respective tensorflow classes
  with tf.GradientTape() as tape:
    prediction = model(input)
    # only using the prediction of the last timestamp
    prediction = prediction[:, -1, 0]
    loss = loss_function(target, prediction)
  gradients = tape.gradient(loss, model.trainable_variables)

  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  return loss

def test(model, test_data, loss_function):
  '''
    Tests the model's performance and calculates loss and accuracy

    Args:
      model: the model in question
      test_data: the test split of the dataset
      loss_function: the loss function to be used

    Returns:
      test_loss: model's loss on the test set
      test_accuracy: model's accuracy on the test set
  '''

  test_accuracy_aggregator = []
  test_loss_aggregator = []

  for (input, target) in test_data:
    prediction = model(input)
    # only using the prediction of the last timestamp
    prediction = prediction[:,-1,0]
    sample_test_loss = loss_function(target, prediction)
    sample_test_accuracy =  target == tf.cast(tf.round(prediction), tf.int32)
    sample_test_accuracy = np.mean(sample_test_accuracy)
    test_loss_aggregator.append(sample_test_loss.numpy())
    test_accuracy_aggregator.append(np.mean(sample_test_accuracy))

  test_loss = tf.reduce_mean(test_loss_aggregator)
  test_accuracy = tf.reduce_mean(test_accuracy_aggregator)

  return test_loss, test_accuracy