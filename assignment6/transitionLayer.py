import tensorflow as tf
import tensorflow.keras as K

class TransitionLayer(K.layers.Layer):

    def __init__(self, number_feature_maps):
        super(TransitionLayer, self).__init__()
        self.conv = K.layers.Conv2D(
            filters = number_feature_maps, kernel_size = 1,
            strides = 2,
            activation = K.activations.relu
            )
        self.batchnorm = K.layers.BatchNormalization()
        self.activation = K.layers.Activation("relu")
        self.average_pooling = K.layers.AvgPool2D(
            pool_size = (2, 2),
            strides = (2, 2)
            )


    def call(self, inputs):
        x = self.conv(inputs)
        x = self.batchnorm(x)
        x = self.activation(x)
        # crashing for unknown reason when including pooling here
        #x = self.average_pooling(x)
        return x
