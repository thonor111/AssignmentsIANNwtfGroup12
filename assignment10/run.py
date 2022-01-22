'''
authors: tnortmann, lmcdonald
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.manifold import TSNE
from input_pipeline import InputPipeline
import training_loop
import tensorflow_text as tf_txt
from skip_gram import SkipGram

with open('dataset/bible.txt') as file:
    data = file.read()

# Hyperparameters
num_epochs = 10
alpha = 0.1
embedding_dimensions = 64
number_vocabulary = 1000

input_pipeline = InputPipeline(data, number_vocabulary=number_vocabulary)
train_data = input_pipeline.prepare_data(data)
# print("IN RUN")
# # print(data)
# print(next(iter(data)))

test_data = input_pipeline.prepare_data_testing("to unto in brought his he")
# print(test_data)
# print(next(iter(test_data)))
print("Created Dataset, start training")

# Initialize Model
model = SkipGram(embedding_dimensions=embedding_dimensions, number_vocabulary=number_vocabulary)


# optimizer
optimizer = K.optimizers.SGD(alpha)

# # initialize lists for later visualization.
train_losses = []
# valid_losses = []
# valid_accuracies = []
#
# # testing once before we begin
# valid_loss, valid_accuracy = training_loop.test(
#     model, test_data, loss_function)
# valid_losses.append(valid_loss)
# valid_accuracies.append(valid_accuracy)
#
# # check how model performs on train data once before we begin
# train_loss, _ = training_loop.test(model, train_data, loss_function)
# train_losses.append(train_loss)
#
# We train for num_epochs epochs.
for epoch in range(num_epochs):
    # training (and checking in with training)
    epoch_losses = []
    for input, target in train_data:
        train_loss = training_loop.train_step(model, input, target, optimizer)
        epoch_losses.append(train_loss)


    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))
    print(f"epoch {epoch} finished with loss of {train_losses[-1]}")

    embeddings = model.embedding_matrix.numpy()
    smallest_loss = 2
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
    best_match = ""
    for elem in test_data:
        embedding_elem = model(elem)
        for i in range(embeddings.shape[0]):
            embedding = tf.nn.embedding_lookup(embeddings, i)
            if cosine_loss(embedding_elem, embedding) < smallest_loss:
                smallest_loss = cosine_loss(embedding_elem, embedding)
                best_match = input_pipeline.words_sorted_shortened[i]
        elem_string = input_pipeline.words_sorted_shortened[elem.numpy()]
        print(f"The most similar element to {elem_string} is {best_match}")
#
#     # testing, so we can track accuracy and test loss
#     valid_loss, valid_accuracy = training_loop.test(model, test_data,
#                                                     loss_function)
#     valid_losses.append(valid_loss)
#     valid_accuracies.append(valid_accuracy)
#
#
# # embedding 1000 testing examples and plotting the latent space
# images_to_embed = test_data.take(1000)
# first_embedding = True
# searching_second_embedding = True
# for input, target, char_class in images_to_embed:
#     embedding = model.encode(input, training = False)
#     embedding = embedding.numpy()
#     if first_embedding:
#         embeddings = embedding.copy()
#         classes = char_class.numpy().copy()
#         first_embedding = False
#         embedding_1 = embedding[0].copy()
#     else:
#         embeddings = np.concatenate((embeddings, embedding), axis = 0)
#         classes = np.concatenate((classes, char_class.numpy()), axis = 0)
#         if searching_second_embedding:
#             embedding_2 = embedding[0].copy()
#             searching_second_embedding = False
#
# embeddings = np.array(embeddings)
# embeddings = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embeddings)
# plt.figure(figsize = (18,12))
# plt.suptitle("Latent Space")
# plt.scatter(embeddings[:,0], embeddings[:,1], c = classes, cmap = 'viridis')
# plt.colorbar()
# plt.show()
#
# plt.figure(figsize=(20,3))
# plt.suptitle("Interpolation between 2 embeddings")
# # plotting the recreations of two embeddings and their interpolations
# for index, i in enumerate(np.linspace(0,1,10)):
#     plt.subplot(1, 10, index+1);
#     interpolation = np.add((1.-i) * embedding_1, i * embedding_2)
#     interpolation = np.reshape(interpolation, (1,10))
#     reconstruction = model.decode(interpolation)
#     reconstruction = np.reshape(reconstruction, (28, 28))
#     plt.imshow(reconstruction)
#     plt.axis("off")
# plt.show()

