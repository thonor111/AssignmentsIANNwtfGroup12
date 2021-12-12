'''
authors: tnortmann, hsanna, lmcdonald
'''

'''
authors: tnortmann, hsanna, lmcdonald
'''

import numpy as np

def integration_task(seq_len, num_samples):
    '''
    Generator function that generates random noise samples sequences between -1
    and 1, and if that sequence sums up to greater or less than 1

    Args:
        seq_len: int that defines the length of the noise sample sequence
        num_samples: int specifiying how many random noise sample sequences to
          generate

    Yields:
        ndarray of random noise sample sequence and target int (0,1)
    '''

    for i in range(num_samples):

        sample = np.random.rand(seq_len) * 2 - 1 # samples between -1 and 1
        sample = np.reshape(sample, (seq_len, 1))
        target = np.sum(sample) >= 1

        yield sample, int(target)

def my_integration_task():
    '''
    Wrapper Generator that calls integration_task() and yields what 
    integration_task() yields. 

    Yields:
        ndarray of random noise sample sequence and target int (0,1)
    '''

    seq_len = 25
    num_samples = 32000

    for sample, target in integration_task(seq_len, num_samples):
        yield sample, target
