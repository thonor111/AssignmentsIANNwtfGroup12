'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

def prepare_datapipeline(dataset):

  #convert data from uint8 to float32
    dataset = dataset.map(lambda img, target: (tf.cast(img, tf.float32), target))
  #sloppy input normalization, just bringing image values from range [0, 127] to [-1, 1]
    dataset = dataset.map(lambda img, target: ((img/127.)-1, target))
  #create one-hot targets
    dataset = dataset.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
  #cache this progress in memory, as there is no need to redo it; it is deterministic after all
    dataset = dataset.cache()
  #shuffle, batch, prefetch
  # shuffle
    dataset= dataset.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    dataset = dataset.batch(64)

    # prefetch
    dataset = dataset.prefetch(20)
    return dataset