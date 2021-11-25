import tensorflow_datasets as tfds
import tensorflow as tf
import numpy as np
from tensorflow.keras import datasets, layers, models

def prepare_mnist_data(mnist):
  
  #convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img/255.), target))
  #create one-hot targets
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    mnist = mnist.cache()
  #shuffle, batch, prefetch
  # shuffle
    mnist= mnist.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    mnist = mnist.batch(10)

    # prefetch
    mnist = mnist.prefetch(20)
    return mnist


    