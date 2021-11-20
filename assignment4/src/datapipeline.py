import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from tensorflow.keras.layers import Dense

def train_validate_test_split(df, train_percent=.7, test_percent=.2):
    '''

    1. takes a pandas dataframe and returns a tensorflow dataset
     splitted in train, test and validation data.
    2. Separates the labels from the input and store them accordingly in each dataset.
    
    Args:

    df: a pandas dataframe
    train_percent:an integer. Default is 0.7
    test_percent: an integer. Default is 0.2

    '''

    #calculating the index for splitting the data 
    m = len(df.index)
    train_end = int(train_percent * m)
    test_end = int(test_percent * m) + train_end

    #splitting the dataset in train, test, validation dataset
    # separeting the labels from the input
    # and create a tensorflow dataset accordingly
    perm = np.random.permutation(df.index)
    train = df.iloc[perm[:train_end]]
    train= (train.drop('quality', axis= 1), train['quality'])
    train= tf.data.Dataset.from_tensor_slices(train)

    test = df.iloc[perm[train_end:test_end]]
    test= (test.drop('quality', axis= 1),  test['quality'])
    test= tf.data.Dataset.from_tensor_slices(test)

    validate = df.iloc[perm[test_end:]]
    validate= (validate.drop('quality', axis=1), validate['quality'])
    validate= tf.data.Dataset.from_tensor_slices(validate)

    return train, test, validate

def make_binary(target):
    '''
    receives a target and returns a target fit 
    for a binary classification task.

    Args: 
    target: a rank 0 tensor

    Returns:
    binary: a rank 0 tensor either 1 or 0
    
    '''
    threshold= tf.cast(6, dtype= 'int64')

    if tf.math.greater(target, threshold):
        binary = tf.cast(1, dtype= 'int64')
    else:
        binary = tf.cast(0, dtype= 'int64')

        
    return binary


def data_pipeline(dataset):
    '''
    take as input a pandas dataset and 
    returns a splitted dataset, with a binary target.

    Args:
    dataset: a pandas dataframe
    '''

    # splitting the dataset in train, test, validation dataset
    train, test, validate= train_validate_test_split(dataset)

    #fitting the target for a binary classification
    train= train.map(lambda input, target: (input, make_binary(target)))
    test= test.map(lambda input, target: (input, make_binary(target)))
    validate= validate.map(lambda input, target: (input, make_binary(target)))

    #saving the datapipeline
    train = train.cache()
    test= test.cache()
    validate= validate.cache()

    #batching 
    train = train.batch(10)
    test = test.batch(10)
    validate= validate.batch(10)


    return train, test, validate


