'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow_datasets as tfds
from data_pipeline import prepare_datapipeline
from utils import classify, visualize, test, train_step, load_data
from ResNet import ResNet

train_ds, valid_ds, test_ds = load_data()

##Testing ResNet model

model = ResNet()

results, trained_model = classify(model, tf.keras.optimizers.Adam(0.001), 30, train_ds, valid_ds)

trained_model.summary()

 _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy())
print("Accuracy (test set):", test_accuracy)

# visualizing losses and accuracy
visualize(results[0],results[1],results[2])

##Testing the model of last week

model_homework = Model()

results_home, trained_model_home = classify(model_homework, tf.keras.optimizers.Adam(0.001), 30, train_ds, valid_ds)

_, test_accuracy = test(trained_model_home, test_ds,tf.keras.losses.CategoricalCrossentropy())
print("Accuracy (test set):", test_accuracy)

# visualizing losses and accuracy
visualize(results_home[0], results_home[1], results_home[2])

