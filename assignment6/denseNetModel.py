'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf
import tensorflow.keras as K
from denseBlock import DenseBlock
from transitionLayer import TransitionLayer


class DenseNetModel(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model implementing the DenseNet structure

    Architecture:
            Feature Learning Layers:
                Convolutional Layer 1:
                Dense Block 1:
                Transition Layer 1:
                Dense Block 2:
                Transition Layer 2:
                Dense Block 3:
                Transition Layer 3:

            Classification Layers:
                Global Pooling Layer 1:
                Output Layer:

    ...

        Attributes
        ----------
        conv1:
        denseBlock1:
        transitionLayer1:
        denseBlock2:
        transitionLayer2:
        denseBlock3:
        transitionLayer3:
        global_pool:
        output_layer :
    '''

    def __init__(self):
        '''
            Initializes hidden and output layers of the model
        '''

        super(DenseNetModel, self).__init__()

        # one convolutional layer to get a greater number of feature maps
        self.conv1 = K.layers.Conv2D(
            filters = 8, kernel_size = 1,
            padding = "same", strides = 1,
            activation = K.activations.relu
            )

        # alternating dense blocks and transition layers increasing and decreasing the number of inputs to the next layer
        self.denseBlock1 = DenseBlock()
        self.transitionLayer1 = TransitionLayer(number_feature_maps = 32)
        self.denseBlock2 = DenseBlock()
        self.transitionLayer2 = TransitionLayer(number_feature_maps=32)
        self.denseBlock3 = DenseBlock()
        self.transitionLayer3 = TransitionLayer(number_feature_maps=32)

        # pooling for classification
        self.global_pool = K.layers.GlobalAveragePooling2D()
        # applying softmax for classification
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
        x = self.denseBlock1(x)
        x = self.transitionLayer1(x)
        x = self.denseBlock2(x)
        x = self.transitionLayer2(x)
        x = self.denseBlock3(x)
        x = self.transitionLayer3(x)
        x = self.global_pool(x)
        y = self.output_layer(x)

        return y