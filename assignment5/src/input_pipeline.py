'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

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

    # input normalization [0,1]
    data = data.map(lambda img, target: ((img/255.), target))

    # create one-hot targets
    data = data.map(lambda img, target: (img, tf.one_hot(target, depth=10)))

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(10)

    # prefetch
    data = data.prefetch(20)

    return data