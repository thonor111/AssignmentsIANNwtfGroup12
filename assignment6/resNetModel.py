'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf
import tensorflow.keras as K
from residualBlock import ResidualBlock


class ResNetModel(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model implementing the ResNet structure

    Architecture:
            Feature Learning Layers:
                Convolutional Layer 1:
                Residual Block 1:
                Convolutional Layer 2:
                Residual Block 2:
            Classification Layers:
                Global Pooling Layer 1:
                Output Layer:

    ...

        Attributes
        ----------
        conv_1:
        residualBlock1:
        conv_2:
        residualBlock2:
        global_pool:
        output_layer :
    '''

    # initialize model with two hidden layers and one output layer
    def __init__(self):
        '''
            Initializes hidden and output layers of the model
        '''

        super(ResNetModel, self).__init__()

        self.conv1 = K.layers.Conv2D(
            filters = 32, kernel_size = 1,
            padding = "same", strides = 1,
            activation = K.activations.relu
            )
        self.residualBlock1 = ResidualBlock(32)
        self.conv2 = K.layers.Conv2D(
            filters=64, kernel_size=1,
            padding="same", strides=1,
            activation=K.activations.relu
        )
        self.residualBlock2 = ResidualBlock(64)
        self.conv2 = K.layers.Conv2D(
            filters=128, kernel_size=1,
            padding="same", strides=1,
            activation=K.activations.relu
        )
        self.residualBlock2 = ResidualBlock(128)

        self.global_pool = K.layers.GlobalAveragePooling2D()

        self.output_layer = K.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs):
        '''
            Forward Step
            Passes activations through the network and calculates prediction

            Args:
                inputs: the inputs to the model

            Returns:
                y: the prediction of the model
        '''

        x = self.conv1(inputs)
        x = self.residualBlock1(x)
        x = self.conv2(x)
        x = self.residualBlock2(x)
        x = self.global_pool(x)
        y = self.output_layer(x)

        return y