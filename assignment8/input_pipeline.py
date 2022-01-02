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
    data = data.map(lambda img, target: (tf.cast(img, tf.float32), target))

    # take the images also as targets
    data = data.map(lambda img, target: (img, img))

    # image normalization [0,1]
    data = data.map(lambda img, target: ((img / 255.) * (1 - noise_strength), target / 255.))

    # adding noise to the image
    data = data.map(lambda img, target: (tf.math.add(img, tf.math.multiply(tf.random.uniform([28,28]), noise_strength)), target))

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(64)

    # prefetch
    data = data.prefetch(20)

    return data