'''
@authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow_datasets as tfds
from data_pipeline import prepare_datapipeline
from utils import classify, visualize, test, train_step, load_data
from ResNet import ResNet

train_ds, valid_ds, test_ds = load_data()

model = ResNet()

results, trained_model = classify(model, tf.keras.optimizers.Adam(0.001), 30, train_ds, valid_ds)

trained_model.summary()

 _, test_accuracy = test(trained_model, test_ds,tf.keras.losses.CategoricalCrossentropy())
print("Accuracy (test set):", test_accuracy)

# visualizing losses and accuracy
visualize(results[0],results[1],results[2])


