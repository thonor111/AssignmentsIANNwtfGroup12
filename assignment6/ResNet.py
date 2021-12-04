'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf

class ResidualBlock(tf.keras.Model):
    def __init__(self, out_filters):
        super(ResidualBlock, self).__init__(name='ResBlock')
        self.conv1 = tf.keras.layers.Conv2D(64, 1, padding='same', input_shape=(32, 32, 3))
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(64, 3, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(out_filters, 1, padding='same')
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.relu = tf.keras.layers.ReLU()
  

    def call(self, x):
        x_1 = self.bn1(x)
        x_1 = self.relu(x_1)
        x_1 = self.conv1(x_1)
        x_1 = self.bn2(x_1)
        x_1 = self.relu(x_1)
        
        x_1 = self.conv2(x_1)
        x_1 = self.bn3(x_1)
        x_1 = self.relu(x_1)
        x_1 = self.conv3(x_1)
        

        x_1 = tf.keras.layers.add([x_1, x])
        return x_1



class ResNet(tf.keras.Model):


    # initialize model with two hidden layers and one output layer
    def __init__(self):
        '''
            Initializes hidden and output layers of the model
        '''

        super(ResNet, self).__init__()
        
        self.conv1= tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding="same", activation="relu")
        self.conv2_1 = ResidualBlock(32)
        self.conv2= tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation="relu")
        self.conv2_2 = ResidualBlock(64)
        self.conv3= tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding="same", activation="relu")
        self.conv2_3 = ResidualBlock(128)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.output_layer = tf.keras.layers.Dense(10, activation = tf.nn.softmax)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2_1(x)
        x = self.conv2(x)
        x = self.conv2_2(x)
        x = self.conv3(x)
        x = self.conv2_3(x)
        x = self.global_pool(x)
        x = self.output_layer(x)
        return x