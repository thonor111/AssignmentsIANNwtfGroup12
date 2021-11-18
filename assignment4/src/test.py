import tensorflow as tf
import tensorflow_datasets as tfds
import input_pipeline, training_loop
from genomics_model import GenomicsModel
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

tf.keras.backend.clear_session()

# load dataset and split into to train (100,000 examples) and test (1,000 examples) sets
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
data = pd.read_csv(url, sep = ';')

train_data, test_data = input_pipeline.create_tf_dataset(data)
train_data = input_pipeline.prepare_data(train_data)
test_data = input_pipeline.prepare_data(test_data)
head = train_data.take(5)
for elem in head:
    print(elem)



#dataset = create_tf_dataset(data)
#dataset = prepare_data(dataset)
#head = dataset.take(5)
#for elem in head:
#    print(elem)

# print(np.array(data).shape)
# inputs = np.array(data)[:, :-1]
# targets = np.array(data["quality"])
# for i in range(inputs.shape[1]):
#     print("column: ", i, "; min: ", inputs[:,i].min(), "; max: ", inputs[:,i].max())
# print(inputs.shape, targets.shape)
