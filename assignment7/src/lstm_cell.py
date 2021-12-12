'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K

class LSTM_Cell(K.layers.Layer):
    '''
    A LSTM Cell with forget, input, output and cell state candidate gates.
    Sublassing K.layers.Layer

    Attributes:
        forget_gate: Dense Layer using sigmoid activation
        input_gate: Dense Layer using sigmoid activation
        cell_state_candidates: Dense Layer using tanh activation
        output_gate: Dense Layer using sigmoid activation
    '''

    def __init__(self, units):
        '''
        Initializes a LSTM Cell.

        Args:
            units: int specifiying the units of the cell
        '''

        super(LSTM_Cell, self).__init__()

        # using an initial bias in the forget gate of 1 as recommended in the 
        # paper by Jozezefowicz et al
        # this helps with keeping a lot of information in th beginning 
        # -> being able to remember and not forget important information
        self.forget_gate = K.layers.Dense(
            units=units, activation="sigmoid",
            bias_initializer=K.initializers.Ones()
        )

        self.input_gate = K.layers.Dense(
            units=units,
            activation=K.activations.sigmoid
        )

        self.cell_state_candidates = K.layers.Dense(
            units=units,
            activation=K.activations.tanh
        )

        self.output_gate = K.layers.Dense(
            units=units,
            activation=K.activations.sigmoid
        )

    @tf.function
    def call(self, x, states):
        '''
        Passes inputs through LSTM Cell, calculates each gate's activations and
        the cell's hidden and cell state.
        
        Args:
            x: input tensor containg input at t
            states: tuple of tensors containig cell and hidden states of t-1

        Returns:
            list containing tensors of hidden state and cell state at t
        '''

        # concatenate hidden state at t-1 and input at t
        hidden_x_concat = tf.concat([states[0], x], axis = 1)

        # calculates activation of forget gate
        forgetting = self.forget_gate(hidden_x_concat)

        # "forgetting" some of the old states
        cell_state = tf.math.multiply(forgetting, states[1])

        # calculating the new candidates of the states
        candidates = self.cell_state_candidates(hidden_x_concat)
        
        # only using some of the candidates
        candidates = tf.math.multiply(
            candidates, self.input_gate(hidden_x_concat))

        # calculates cell state at t
        cell_state = tf.math.add(cell_state, candidates)

        # calculates activation at output gate
        out = self.output_gate(hidden_x_concat)

        # calculates hidden state at t
        hidden = tf.math.multiply(K.activations.tanh(cell_state), out)

        return [hidden, cell_state]