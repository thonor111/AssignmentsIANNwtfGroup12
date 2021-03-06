'''
authors: tnortmann, hsanna, lmcdonald
'''

import numpy as np
import tensorflow as tf


def train_step(generator, discriminator, images, noises, loss_function, optimizer_generator, optimizer_discriminator):
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
    with tf.GradientTape() as discriminator_tape:
        generation = generator(noises[0], training=True)
        prediction_fake = discriminator(generation, training=True)
        prediction_real = discriminator(images, training=True)
        loss_discriminator_fake = loss_function(tf.zeros_like(prediction_fake), prediction_fake)
        loss_discriminator_real = loss_function(tf.ones_like(prediction_real), prediction_real)
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
    # training the discriminator
    gradients_discriminator = discriminator_tape.gradient(loss_discriminator, discriminator.trainable_variables)
    optimizer_discriminator.apply_gradients(zip(gradients_discriminator, discriminator.trainable_variables))
    # calculating the loss of the generator with the updated discriminator
    for noise in noises:
        with tf.GradientTape() as generator_tape:
            generation = generator(noise, training=True)
            prediction_fake = discriminator(generation, training=True)
            loss_generator = loss_function(tf.ones_like(prediction_fake), prediction_fake)
            # training the generator
        gradients_generator = generator_tape.gradient(loss_generator, generator.trainable_variables)
        optimizer_generator.apply_gradients(zip(gradients_generator, generator.trainable_variables))

    return (loss_discriminator, loss_generator)

def train_step_generator(generator, discriminator, noises, loss_function, optimizer_generator):
    '''
    Performs the training step only on the generator to work against a too strong discriminator in comparison to the
    Generator, resulting in vanishing gradients

    Args:
      model: the model to be trained
      input: the input data
      target: the targets corresponding to the input data
      loss_function: the loss_function to be used
      optimizer: the optimizer to be used

    Returns:
      loss: the loss of the current epoch
  '''

    for noise in noises:
        with tf.GradientTape() as generator_tape:
            generation = generator(noise, training=True)
            prediction_fake = discriminator(generation, training=True)
            loss_generator = loss_function(tf.ones_like(prediction_fake), prediction_fake)
            # training the generator
        gradients_generator = generator_tape.gradient(loss_generator, generator.trainable_variables)
        optimizer_generator.apply_gradients(zip(gradients_generator, generator.trainable_variables))


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

    test_loss_aggregator_generator = []
    test_loss_aggregator_discriminator = []

    for (images, noises) in test_data:
        generation = generator(noises[0], training=True)
        prediction_fake = discriminator(generation, training=True)
        prediction_real = discriminator(images, training=True)
        loss_discriminator_fake = loss_function(tf.zeros_like(prediction_fake), prediction_fake)
        loss_discriminator_real = loss_function(tf.ones_like(prediction_real), prediction_real)
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        loss_generator = loss_function(tf.ones_like(prediction_fake), prediction_fake)
        test_loss_aggregator_generator.append(loss_generator.numpy())
        test_loss_aggregator_discriminator.append(loss_discriminator.numpy())

    test_loss_generator = tf.reduce_mean(test_loss_aggregator_generator)
    test_loss_discriminator = tf.reduce_mean(test_loss_aggregator_discriminator)

    return (test_loss_discriminator, test_loss_generator)
