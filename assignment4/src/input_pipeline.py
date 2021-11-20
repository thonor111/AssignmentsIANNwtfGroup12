'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import numpy as np


def make_binary(target):
    '''
    transforms the given metric target to binary values
    '''
    # according to the website we got the dataset from the target can have values between 0 and 10 so we used 5 as a threshold
    # actually, this dataset only has values from 3 to 8 so we included 5 in the values mapped to 0
    return int(target > 5)

def unify(input):
    output = []
    for column_index in range(input.shape[1]):
        column = input[:, column_index]
        column = np.divide((column - column.min()), (column.max() - column.min()))
        output.append(column)
    return np.array(output).T

def create_tf_dataset(data):
    '''
    splits the given data into input and target and turns it into a tf dataset
    '''
    # every column but the last are inputs (turning it to a numpy array automatically removes the indices and labels)
    inputs = np.array(data)[:, :-1]

    inputs = unify(inputs)

    # The last column, "quality", is our target -> We want to learn which wine has a good quality
    targets = np.array(data["quality"])

    # creating the tf dataset
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    # shuffling before splitting to have all targets in training and testing dataset
    dataset = dataset.shuffle(1000)

    # splitting training and testing set
    training_set = dataset.take(1400)
    dataset = dataset.skip(1400)
    testing_set = dataset.take(199)


    return training_set, testing_set

def prepare_data(data):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data

        Args:
            data: tf dataset

        Returns:
            data: the prepared tf dataset
    '''

    # make binary
    data = data.map(lambda input, target: (input, make_binary(target)))

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True)

    # cache
    data = data.cache()

    # batch
    data = data.batch(50)

    # prefetch
    data = data.prefetch(20)

    return data