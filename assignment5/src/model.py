'''
@authors: tnortmann, hsanna, lmcdonald
'''
import tensorflow as tf
import tensorflow.keras as K

class Model(tf.keras.Model):
    '''
    Custom Model subclassing from tf.keras.Model

    Architecture:  
            Feature Learning Layers:
                Convolutional Layer 1: 
                Convolutional Layer 2: 
                Convolutional Layer 3:
                Max Pooling Layer 1:
                Max Pooling Layer 2:
            Classification Layers:
                Global Pooling Layer 1:
                Output Layer:     

    ...

        Attributes
        ----------
        conv_1:
        conv_2:
        conv_3:
        max_pool_1:
        max_pool_2:
        global_pool:
        output_layer : 
    '''

    # initialize model with two hidden layers and one output layer
    def __init__(self):
        '''
            Initializes hidden and output layers of the model
        '''

        super(Model, self).__init__()
        
        self.conv_1 = K.layers.Conv2D(
            filters = 8, kernel_size = (3,3),
            padding = "same", strides = (1,1),
            activation = K.activations.relu
            )

        self.max_pool_1 = K.layers.MaxPool2D()

        self.conv_2 = K.layers.Conv2D(
            filters = 16, kernel_size = (3,3),
            padding = "same", strides = (1,1),
            activation = K.activations.relu
            )
        
        self.max_pool_2 = K.layers.MaxPool2D()

        self.conv_3 = K.layers.Conv2D(
            filters = 32, kernel_size = (3,3),
            padding = "same", strides = (1,1),
            activation = K.activations.relu
            )

        self.global_pool = K.layers.GlobalAveragePooling2D()

        self.output_layer = K.layers.Dense(10, activation = tf.nn.softmax)

    def call(self, inputs):
        '''
            Forward Step
            Passes activations through the network and calculates prediction

            Args:
                inputs: the inputs to the model

            Returns:
                y: the prediction of the model
        '''

        x = self.conv_1(inputs)
        x = self.max_pool_1(x)
        x = self.conv_2(x)
        x = self.max_pool_2(x)
        x = self.conv_3(x)
        x = self.global_pool(x)
        y = self.output_layer(x)

        return y