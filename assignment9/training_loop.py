'''
authors: tnortmann, hsanna, lmcdonald
'''

import numpy as np
import tensorflow as tf


def train_step(generator, discriminator, images, noise, loss_function, optimizer):
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
  with tf.GradientTape() as generator_tape:
    with tf.GradientTape() as discriminator_tape:
      generation = generator(noise, training = True)
      prediction_fake = discriminator(generation, training = True)
      prediction_real = discriminator(images, training = True)
      loss_discriminator_fake = loss_function(tf.zeros_like(prediction_fake), prediction_fake)
      loss_discriminator_real = loss_function(tf.ones_like(prediction_real), prediction_real)
      loss_discriminator = loss_discriminator_real + loss_discriminator_fake
      loss_generator = loss_function(tf.ones_like(prediction_fake), prediction_fake)
  gradients_discriminator = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)
  gradients_generator = generator_tape.gradient(loss_generator, generator.trainable_variables)

  optimizer.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
  optimizer.apply_gradients(zip(gradients_generator, generator.trainable_variables))

  return loss_discriminator + loss_generator

def test(generator, discriminator, test_data, loss_function):
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

  test_loss_aggregator = []

  for (images, noise) in test_data:
    generation = generator(noise, training=True)
    prediction_fake = discriminator(generation, training=True)
    prediction_real = discriminator(images, training=True)
    loss_discriminator_fake = loss_function(tf.zeros_like(prediction_fake), prediction_fake)
    loss_discriminator_real = loss_function(tf.ones_like(prediction_real), prediction_real)
    loss_discriminator = loss_discriminator_real + loss_discriminator_fake
    loss_generator = loss_function(tf.ones_like(prediction_fake), prediction_fake)
    sample_test_loss = loss_discriminator + loss_generator
    test_loss_aggregator.append(sample_test_loss.numpy())

  test_loss = tf.reduce_mean(test_loss_aggregator)

  return test_loss