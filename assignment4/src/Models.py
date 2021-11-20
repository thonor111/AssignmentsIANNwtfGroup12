import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Dense

'''
Creating two models:
MyModel(): baseline model

MyModel_dropout(): model with dropout layers
'''


class MyModel(tf.keras.Model):
    
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)

    
    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.out(x)
        return x


class MyModel_dropout(tf.keras.Model):
    
    def __init__(self):
        super(MyModel_dropout, self).__init__()
        self.dense1 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.sigmoid)
        self.out = tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)
        self.dropout_layer = tf.keras.layers.Dropout(rate=0.2)

    
    def call(self, inputs, training= True):
        x = self.dense1(inputs)
        x= self.dropout_layer(x)
        x = self.dense2(x)
        x= self.dropout_layer(x)
        x = self.out(x)
        return x