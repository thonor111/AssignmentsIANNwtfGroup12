import tensorflow as tf
import tensorflow_datasets as tfds
import input_pipeline, training_loop
from resNetModel import ResNetModel
from denseNetModel import DenseNetModel
import matplotlib.pyplot as plt
import tensorflow.keras as K

tf.keras.backend.clear_session()

# load and split dataset into train, validation and test split
train_data, valid_data, test_data = tfds.load('cifar10', split =
                ['train', 'train', 'test'], as_supervised = True)

# apply input pipeline to dataset splits
train_data = train_data.apply(input_pipeline.prepare_data)
valid_data = valid_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

# Hyperparameters
num_epochs = 10
alpha = 0.1

# Initialize Models
#resNetModel = ResNetModel()
resNetModel = DenseNetModel()

# loss function
loss_function = K.losses.CategoricalCrossentropy()

# optimizer
optimizer = K.optimizers.SGD(alpha)

# initialize lists for later visualization.
train_losses_res = []
valid_losses_res = []
valid_accuracies_res = []

# testing once before we begin
valid_loss_res, valid_accuracy_res = training_loop.test(resNetModel, test_data, loss_function)
valid_losses_res.append(valid_loss_res)
valid_accuracies_res.append(valid_accuracy_res)

# check how model performs on train data once before we begin
train_loss_res, _ = training_loop.test(resNetModel, train_data, loss_function)
train_losses_res.append(train_loss_res)

# We train for num_epochs epochs.
for epoch in range(num_epochs):

    # print out starting accuracy
    print(f'Epoch: {str(epoch)} starting with accuracy {valid_accuracies_res[-1]}')

    # training (and checking in with training)
    epoch_losses_res = []
    for input, target in train_data:
        train_loss_res = training_loop.train_step(resNetModel, input, target,
                                        loss_function, optimizer)
        epoch_losses_res.append(train_loss_res)

    # track training loss
    train_losses_res.append(tf.reduce_mean(epoch_losses_res))

    # testing, so we can track accuracy and test loss
    valid_loss_res, valid_accuracy_res = training_loop.test(resNetModel, valid_data,
                                                loss_function)
    valid_losses_res.append(valid_loss_res)
    valid_accuracies_res.append(valid_accuracy_res)

# Visualize accuracy and loss for training and test data.
plt.figure()
line1, = plt.plot(train_losses_res)
line2, = plt.plot(valid_losses_res)
line3, = plt.plot(valid_accuracies_res)
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.ylim(top = 1)
plt.ylim(bottom = 0)
plt.legend((line1,line2, line3),("training","test", "test accuracy"))
plt.show()


