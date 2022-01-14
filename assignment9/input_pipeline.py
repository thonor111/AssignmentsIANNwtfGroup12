'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

noise_strength = 0.2

def prepare_data(data):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data

        Args:
            data: tf dataset

        Returns:
            data: the prepared tf dataset
    '''

    # convert data from uint8 to float32
    data = data.map(lambda img: tf.cast(img, tf.float32))

    # image normalization [-1,1]
    data = data.map(lambda img: img / 128 - 1)

    # reshaping
    data = data.map(lambda img: tf.reshape(img, (28,28,1)))

    # adding noise to the image
    data = data.map(lambda img: (img, tf.random.uniform(shape=[100])))

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(64)

    # prefetch
    data = data.prefetch(20)

    return data