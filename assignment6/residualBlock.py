'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf
import tensorflow.keras as K

class ResidualBlock(K.layers.Layer):


    def __init__(self, input_channel_dimension):
        '''
        Initializes the residual block given the number of channels of the input and the output
        :param input_channel_dimension: the number of channels of the input to make the output the same size
        '''
        super(ResidualBlock, self).__init__()
        # alternation of convolution layer, batchnorm and activation function
        self.conv1 = K.layers.Conv2D(
            filters = 64, kernel_size = 1,
            padding = "same", strides = 1
            )
        self.batchnorm1 = K.layers.BatchNormalization()
        self.activation = K.layers.Activation("relu")
        self.conv2 = K.layers.Conv2D(
            filters=64, kernel_size=3,
            padding="same", strides=1
        )
        self.batchnorm2 = K.layers.BatchNormalization()
        self.conv3 = K.layers.Conv2D(
            filters=input_channel_dimension, kernel_size=1,
            padding="same", strides=1
        )
        self.batchnorm3 = K.layers.BatchNormalization()



    def call(self, inputs):
        '''
        forward step through the residual block
        :param inputs: the inputs of the block
        :return: the processed data to be fed into the next block or layer
        '''
        x = self.conv1(inputs)
        x = self.batchnorm1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        # adding the inputs to the result of this block
        x = tf.add(x, inputs)
        return x