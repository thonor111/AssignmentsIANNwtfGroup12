'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K
from lstm_cell import LSTM_Cell

class LSTM_Layer(K.layers.Layer):
    '''
    A LSTM Layer containing at least on LSTM Cell.
    Subclassing from K.layers.Layer

    Attributes:
        cell_depth: int specifiying size of cell
        cell: LSTM cell of depth cell_depth
    '''

    def __init__(self):
        '''
        Initializes a LSTM layer.
        '''

        super(LSTM_Layer, self).__init__()

        self.cell_depth = 8
        self.cell = LSTM_Cell(self.cell_depth)

    @tf.function
    def call(self, x, states):
        '''
        Unrolling LSTM layer, calculating states at each timestep using 
        Graph Mode.

        Args:
            x: input tensor containg input at t
            states: tuple of tensors containig cell and hidden states of t-1

        Returns:
            Activation tensor for each timestep

        '''

        outputs = tf.TensorArray(tf.float32, size=0, dynamic_size=True)
        x_unstacked = tf.unstack(x, axis = 1)

        for index in range(tf.shape(x)[1]):

            # equivalent to elem = x_unstacked[index], 
            # just compatible with graph-mode
            elem = tf.gather(x_unstacked, index)
            states = self.cell(elem, states)
            outputs = outputs.write(index, states[0])

        return outputs.stack()

    def zero_states(self, batch_size):
        '''
        Initializes cell and hidden states at timestep n with zeros.

        Args:
            batch_size: int specifiying amount of inputs

        Returns:
            list of tensors containing the zero-initialized cell and hidden 
              states of size batch_size
        '''
        
        cell_states =  tf.zeros((batch_size, self.cell_depth))
        hidden_states = tf.zeros((batch_size, self.cell_depth))

        return [hidden_states, cell_states]