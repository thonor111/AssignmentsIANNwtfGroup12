'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from lstm_cell import LSTM_Cell

class LSTM_Layer:

    def __init__(self):
        self.cell = LSTM_Cell(20)

    def call(self, x, states):
