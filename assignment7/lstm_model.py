import tensorflow as tf
import tensorflow.keras as K
from lstm_layer import LSTM_Layer

class LSTM_Model(K.Model):

    def __init__(self):
        super(LSTM_Model, self).__init__()
        self.read_in = K.layers.Dense(8, activation = "sigmoid")
        self.lstm = LSTM_Layer()
        self.output_layer = K.layers.Dense(1, activation = "sigmoid")

    def call(self, x):
        x = self.read_in(x)
        x = self.lstm(x, self.lstm.zero_states(64))
        x = self.output_layer(x)
        return x