'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow.keras as K

class LSTM_Cell(K.layers.Layer):

    def __init__(self, units):
        super(LSTM_Cell, self).__init__()
        # using an initial bias in the forget gate of 1 as recommended in the paper by Jozezefowicz et al
        # this helps with keeping a lot of information in th beginning -> being able to remember and not forget important information
        self.forget_gate = K.layers.Dense(units = units, activation = "sigmoid", bias_initializer=K.initializers.Ones())
        self.input_gate = K.layers.Dense(units = units, activation = "sigmoid")
        self.cell_state_candidates = K.layers.Dense(units = units, activation = "tanh")
        self.output_gate = K.layers.Dense(units = units, activation = "sigmoid")
        self.tanh = K.layers.Activation("tanh")

    @tf.function
    def call(self, x, states):
        '''
        :param x:
        :param states: tupel (hidden state, cell state)
        :return:
        '''
        # where to forget
        forgetting = self.forget_gate(tf.concat([states[0], x], axis = 1))
        # "forgetting" some of the old states
        new_cell_state = tf.math.multiply(forgetting, states[1])
        # calculating the new candidates of the states
        candidates = self.cell_state_candidates(tf.concat([states[0], x], axis = 1))
        # only using some of the candidates
        candidates = tf.math.multiply(candidates, self.input_gate(tf.concat([states[0], x], axis = 1)))
        new_cell_state = tf.math.add(new_cell_state, candidates)
        out = self.output_gate(tf.concat([states[0], x], axis = 1))
        out = tf.math.multiply(self.tanh(new_cell_state), out)
        return out, [out, new_cell_state]