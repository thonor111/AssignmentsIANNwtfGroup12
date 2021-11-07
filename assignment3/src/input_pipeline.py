'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

def genome_onehot(tensor):
    '''
    
    '''

    # dictionary that maps each base in genome string to an int value between 0 and 3
    base_value_map = {"A" : "0", "C" : "1", "G" : "2", "T" : "3"}

    # replace base chars with value ints
    for base, value in base_value_map.items():
        tensor = tf.strings.regex_replace(tensor, base, value)

    split = tf.strings.bytes_split(tensor)
    labels = tf.cast(tf.strings.to_number(split), tf.uint8)
    onehot = tf.one_hot(labels, 4)
    onehot = tf.reshape(onehot, (-1,))
    return onehot

def prepare_data(data):
    '''
    '''

    # turn sequence and labels into one-hot vectors
    data = data.map(lambda x,y: (genome_onehot(x),tf.reshape(tf.one_hot(y, depth = 10), (-1,))))

    # cache
    data = data.cache()

    # shuffle
    data = data.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    data = data.batch(10)

    # prefetch
    data = data.prefetch(20)

    return data