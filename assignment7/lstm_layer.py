'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from lstm_cell import LSTM_Cell

class LSTM_Layer(K.layers.Layer):

    def __init__(self):
        super(LSTM_Layer, self).__init__()
        self.cell_depth = 8
        self.cell = LSTM_Cell(self.cell_depth)

    def call(self, x, states):
        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        x_unstacked = tf.unstack(x, axis = 1)
        for index in range(tf.shape(x)[1]):
            elem = x_unstacked[index]
            out, states = self.cell(elem, states)
            outputs = outputs.write(index, out)
        # transposing to divide by entries and not by batch-elements
        # for i, elem in enumerate(tf.transpose(x)):
        #     out, states = self.cell(elem, states)
        #     outputs = outputs.write(i, out)
        return outputs.stack()

    def zero_states(self, batch_size):
        cell_states =  tf.zeros((batch_size, self.cell_depth))
        hidden_states = tf.zeros((batch_size, self.cell_depth))
        return [hidden_states, cell_states]