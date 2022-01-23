import tensorflow.keras as K
import tensorflow as tf

class SkipGram(K.layers.Layer):

    def __init__(self, embedding_dimensions = 64, number_vocabulary = 10000):
        super(SkipGram, self).__init__()
        self.embedding_dimensions = embedding_dimensions
        self.number_vocabulary = number_vocabulary

    def build(self, input_shape):
        self.embedding_matrix = self.add_weight('embedding_matrix', (self.number_vocabulary, self.embedding_dimensions),
                                                trainable=True, initializer="random_normal",
                                                dtype="float32")

    def call(self, inputs):
        embedding = tf.nn.embedding_lookup(self.embedding_matrix, inputs)
        return embedding
