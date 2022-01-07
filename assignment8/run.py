'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from sklearn.manifold import TSNE
import input_pipeline
import training_loop
from autoencoder import Autoencoder

train_data, test_data = tfds.load('mnist', split=['train', 'test'], as_supervised=True)


# prepare data
train_data = train_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1
embedding_dimensions = 10

# Initialize Model
model = Autoencoder(embedding_dimensions)

# loss function
loss_function = K.losses.MeanSquaredError()

# optimizer
optimizer = K.optimizers.SGD(alpha)

# initialize lists for later visualization.
train_losses = []
valid_losses = []
valid_accuracies = []

# testing once before we begin
valid_loss, valid_accuracy = training_loop.test(
    model, test_data, loss_function)
valid_losses.append(valid_loss)
valid_accuracies.append(valid_accuracy)

# check how model performs on train data once before we begin
train_loss, _ = training_loop.test(model, train_data, loss_function)
train_losses.append(train_loss)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # print out starting accuracy
    print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies[-1]}')

    # plotting some testing examples
    plotting_examples = test_data.take(3)
    for input, target, char_class in plotting_examples:
        plt.gray()
        plt.axis("off")
        plt.subplot(1, 2, 1);
        plt.imshow(input.numpy()[0].reshape((28,28)))
        plt.title("Input")
        plt.subplot(1, 2, 2);
        plt.imshow(model(input).numpy()[0].reshape((28,28)))
        plt.title("Reconstruction")
        plt.show()

    # training (and checking in with training)
    epoch_losses = []
    for input, target, char_class in train_data:
        train_loss = training_loop.train_step(model, input, target,
                                              loss_function, optimizer)
        epoch_losses.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))

    # testing, so we can track accuracy and test loss
    valid_loss, valid_accuracy = training_loop.test(model, test_data,
                                                    loss_function)
    valid_losses.append(valid_loss)
    valid_accuracies.append(valid_accuracy)

# Visualize accuracy and loss for training and test data.
# plt.figure()
# line1 = plt.plot(valid_accuracies)
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.ylim(top=1)
# plt.ylim(bottom=0)
# plt.legend([line1], ["Validation accuracy"])
# plt.show()


images_to_embed = test_data.take(1000)
first_embedding = True
for input, target, char_class in images_to_embed:
    embedding = model.encode(input)
    embedding = embedding.numpy()
    embedding = TSNE(n_components=2, learning_rate='auto', init='random').fit_transform(embedding)
    if first_embedding:
        embeddings = embedding.copy()
        classes = char_class.numpy().copy()
        first_embedding = False
    else:
        embeddings = np.concatenate((embeddings, embedding), axis = 0)
        #print(embeddings.shape)
        classes = np.concatenate((classes, char_class.numpy()), axis = 0)
embeddings = np.array(embeddings)
plt.figure()
plt.scatter(embeddings[:,0], embeddings[:,1], c = classes, cmap = 'viridis')
plt.colorbar()
plt.show()

