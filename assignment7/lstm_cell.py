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
        #self.forget_gate = K.layers.Dense(units=units, activation="sigmoid")
        self.input_gate = K.layers.Dense(units = units, activation = "sigmoid")
        self.cell_state_candidates = K.layers.Dense(units = units, activation = "tanh")
        self.output_gate = K.layers.Dense(units = units, activation = "sigmoid")
        self.tanh = K.layers.Activation("tanh")

    def call(self, x, states):
        # "forgetting" some of the old states
        # where to forget
        forgetting = self.forget_gate(x)
        states = tf.math.multiply(forgetting, states)
        # calculating the new candidates ofr the states
        candidates = self.cell_state_candidates(x)
        # only using some of the candidates
        candidates = tf.math.multiply(candidates, self.input_gate(x))
        states = tf.math.add(states, candidates)
        out = self.output_gate(x)
        out = tf.math.multiply(self.tanh(states), out)
        return out, states