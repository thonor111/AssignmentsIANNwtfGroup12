'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from lstm_cell import LSTM_Cell

class LSTM_Layer(K.layers.Layer):

    def __init__(self):
        super(LSTM_Layer, self).__init__()
        self.cell = LSTM_Cell(64)

    def call(self, x, states):
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        # transposing to divide by entries and not by batch-elements
        for i, elem in enumerate(tf.transpose(x)):
            out, states = self.cell(elem, states)
            outputs = outputs.write(i, out)
        return outputs.stack()

    def zero_states(self, batch_size):
        return tf.zeros(batch_size)