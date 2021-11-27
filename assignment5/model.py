import tensorflow as tf
import tensorflow.keras as keras

class Model(keras.Model):

    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = keras.layers.Conv2D(
            filters = 8, kernel_size = (3,3),
            padding = "same", strides = (1,1),
            activation = "relu"
            )
        self.conv2 = keras.layers.Conv2D(
            filters=16, kernel_size=(3, 3),
            padding="same", strides=(1, 1),
            activation="relu"
            )
        self.conv3 = keras.layers.Conv2D(
            filters=32, kernel_size=(3, 3),
            padding="same", strides=(1, 1),
            activation="relu"
            )

        self.max_pool_1 = keras.layers.MaxPool2D()
        self.max_pool_2 = keras.layers.MaxPool2D()
        self.global_pool = keras.layers.GlobalAvgPool2D()

        self.out = keras.layers.Dense(10, activation = "softmax")


    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.max_pool_1(x)
        x = self.conv2(x)
        x = self.max_pool_2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.out(x)
        return x