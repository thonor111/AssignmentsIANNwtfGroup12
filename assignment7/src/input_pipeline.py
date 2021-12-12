'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

def prepare_data(data):
    '''
        Prepares a tf dataset
        Caches, shuffles, batches and prefetches data

        Args:
            data: tf dataset

        Returns:
            data: the prepared tf dataset
    '''

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(64)

    # prefetch
    data = data.prefetch(20)

    return data