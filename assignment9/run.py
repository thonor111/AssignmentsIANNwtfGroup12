'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import input_pipeline
import training_loop
import urllib
import os
from generator import Generator
from discriminator import Discriminator


category = 'candle'

if not os.path.isdir('dataset'):
    os.mkdir('dataset')

if not os.path.exists(f'dataset/{category}.npy'):
    url = f'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{category}.npy'
    urllib.request.urlretrieve(url, f'dataset/{category}.npy')

images = np.load(f'dataset/{category}.npy')
print(f'{len(images)} images to train on')

# You can limit the amount of images you use for training by setting :
train_images = images[:10000]
test_images = images[10000:11000]
# plt.gray()
# plt.imshow(train_images[0].reshape((28,28)))
# plt.show()

# create tf datasets
train_data = tf.data.Dataset.from_tensor_slices(train_images)
test_data = tf.data.Dataset.from_tensor_slices(test_images)

# prepare data
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.001

# Initialize Model
generator = Generator()
discriminator = Discriminator()

# loss function
loss_function = K.losses.BinaryCrossentropy()

# optimizer
optimizer = K.optimizers.Adam(alpha)

# initialize lists for later visualization.
train_losses = []
test_losses = []

# testing once before we begin
test_loss = training_loop.test(
    generator, discriminator, test_data, loss_function)
test_losses.append(test_loss)

# check how model performs on train data once before we begin
train_loss = training_loop.test(generator, discriminator, train_data, loss_function)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # visualizing the generator
    plotting_examples = test_data.take(5)
    plot_number = 1
    plt.figure(figsize=(20, 5))
    plt.suptitle(f"Epoch {epoch} with a starting test-loss of {test_losses[-1]}")
    for image, noise in plotting_examples:
        plt.subplot(1, 5, plot_number)
        plt.gray()
        plt.imshow(generator(noise).numpy()[0].reshape((28, 28)))
        plt.title(f"Candle {plot_number}")
        plt.axis("off")
        plot_number += 1
    plt.show()


    # training (and checking in with training)
    epoch_losses = []
    for image, noise in train_data:
        train_loss = training_loop.train_step(generator, discriminator, image, noise,
                                              loss_function, optimizer)
        epoch_losses.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))

    # testing, so we can track accuracy and test loss
    test_loss = training_loop.test(generator, discriminator, test_data,
                                                    loss_function)
    test_losses.append(test_loss)

plt.figure()
line1, = plt.plot(train_losses)
line2, = plt.plot(test_losses)
plt.xlabel("Epoch")
plt.ylabel("Losses")
plt.ylim(bottom = 0)
plt.legend((line1,line2),("Train losses","Test losses"))
plt.show()

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

