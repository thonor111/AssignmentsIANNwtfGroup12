'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

def make_binary(target):
    '''
        Transforms the target value (1 - 10 in wine dataset) into binary target
        (0: worse than average = bad, 1: better than average = good). 
        Average is used instead of median, because if you had a dataset of five
        wines with labels 10 2 1 1 1, median would be 1, so the quality 2 wine
        would be labeld as good, while the average would be 3, so the quality
        2 wine would be labeled as bad, which makes more sense. Thus the average
        better represents the placement between highest and lowest quality.

        Args:
            

        Returns:
            
    '''
    #mean = target.reduce(0., tf.math.add)/ tf.cast(target.cardinality(),tf.float32)
    if target < 6:
        return 0
    else:
        return 1

def prepare_data(data):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data

        Args:
            data: tf dataset

        Returns:
            data: the prepared tf dataset
    '''

    #
    data = data.map(lambda input, target: (input, make_binary(target[0])))

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(10)

    # prefetch
    data = data.prefetch(20)

    return data