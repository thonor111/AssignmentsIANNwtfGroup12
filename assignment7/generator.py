'''
authors: tnortmann, hsanna, lmcdonald
'''

import numpy as np

def integration_task(seq_len, num_samples):
    for i in range(num_samples):
        data_points = np.random.rand(seq_len) * 2 - 1
        target = np.sum(data_points) >= 0
        yield data_points, int(target)

def my_integration_task():
    seq_len = 10
    num_samples = 1000
    for elem in integration_task(seq_len, num_samples):
        yield elem
