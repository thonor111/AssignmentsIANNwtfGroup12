import tensorflow as tf
from SimpleDense import SimpleDense

class MyModel(tf.keras.Model):

    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = SimpleDense(units=256)
        self.dense2 = SimpleDense(units=256)
        self.out = SimpleDense(units=10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x