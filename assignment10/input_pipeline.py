import tensorflow as tf

def skip_gram_generator(all_tokens, vocab, window_size):

  for i, token in enumerate(all_tokens):
    if token in vocab:
      j = - int(window_size/2)
      while j <= int(window_size/2):
        if j != 0 and i + j in range(len(all_tokens)):
          yield token, all_tokens[i + j]
        j+= 1

def prepare_data(data):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data
        Args:
            data: tf dataset
        Returns:
            data: the prepared tf dataset
    '''

    data = data.map(lambda input, target: (tf.cast(input, tf.int64), tf.cast(target, tf.int64)))

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(64)

    # prefetch
    data = data.prefetch(20)

    return data

def prepare_data_test(data):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data
        Args:
            data: tf dataset
        Returns:
            data: the prepared tf dataset
    '''

    data = data.map(lambda input: tf.cast(input, tf.int64))

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # prefetch
    data = data.prefetch(20)

    return data