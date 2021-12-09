from generator import my_integration_task
import tensorflow as tf
import input_pipeline


dataset = tf.data.Dataset.from_generator(my_integration_task, output_types=tf.float32)

number_data_points = 1000

train_data = dataset.take(int(0.9 * number_data_points))
valid_data = dataset.take(int(0.9 * number_data_points))
dataset.skip(int(0.9 * number_data_points))
test_data = dataset.take(int(0.1 * number_data_points))

train_data = train_data.apply(input_pipeline.prepare_data)
valid_data = valid_data.apply(input_pipeline.prepare_data)
test_data = test_data.apply(input_pipeline.prepare_data)

