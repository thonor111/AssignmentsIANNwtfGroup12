'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow.keras as K

class DenseBlock(K.layers.Layer):

    def __init__(self):
        '''
        initializes the dense block with convolutional, batchnorm and activation layers
        '''
        super(DenseBlock, self).__init__()
        self.conv1 = K.layers.Conv2D(
            filters=128, kernel_size=1,
            padding="same", strides=1
        )
        self.batchnorm1 = K.layers.BatchNormalization()
        self.activation = K.layers.Activation("relu")
        self.conv2 = K.layers.Conv2D(
            filters=32, kernel_size=3,
            padding="same", strides=1
        )
        self.batchnorm2 = K.layers.BatchNormalization()



    def call(self, inputs):
        '''
        forward step through the dense block
        :param inputs: the inputs of the block
        :return: the processed data to be fed into the next block or layer
        '''
        x = self.batchnorm1(inputs)
        x = self.activation(x)
        x = self.conv1(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        # concatenating the result of the block and the inputs of the block
        x = K.layers.concatenate([x, inputs])
        return x