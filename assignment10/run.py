'''
authors: tnortmann, lmcdonald
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
from input_pipeline import InputPipeline
import training_loop
from skip_gram import SkipGram

with open('dataset/bible.txt') as file:
    data = file.read()

# Hyperparameters
num_epochs = 10
alpha = 0.1
embedding_dimensions = 64
number_vocabulary = 100000

input_pipeline = InputPipeline(data, number_vocabulary=number_vocabulary)
train_data = input_pipeline.prepare_data(data)

test_data = input_pipeline.prepare_data_testing("holy father wine poison love strong day")
print("Created Dataset, start training")

# Initialize Model
model = SkipGram(embedding_dimensions=embedding_dimensions, number_vocabulary=number_vocabulary)


# optimizer
optimizer = K.optimizers.SGD(alpha)

# # initialize lists for later visualization.
train_losses = []

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    # training (and checking in with training)
    epoch_losses = []
    for input, target in train_data:
        train_loss = training_loop.train_step(model, input, target, optimizer, number_vocabulary=number_vocabulary)
        epoch_losses.append(train_loss)


    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))
    print(f"epoch {epoch} finished with loss of {train_losses[-1]}")

    k = 5
    embeddings = model.embedding_matrix.numpy()
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
    best_match = ""
    for elem in test_data:
        embedding_elem = model(elem)
        cosine_losses = np.ones(embeddings.shape[0])
        for i in range(embeddings.shape[0]):
            embedding = tf.nn.embedding_lookup(embeddings, i)
            cosine_losses[i] = cosine_loss(embedding_elem, embedding)
        indices = np.argsort(cosine_losses)
        indices = indices[:k]
        best_matches = ""
        for j in range(k):
            best_matches += str(input_pipeline.words_sorted_shortened[indices[j]]) + " "
        elem_string = input_pipeline.words_sorted_shortened[elem.numpy()]
        print(f"The most similar elements to {elem_string} are {best_matches}")


