'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow_text as tf_txt
import numpy as np


class InputPipeline:

    def __init__(self, text):
        self.number_words = 1000

        data = self.create_data(text)

        # getting the counts of the different words
        first_elem = True
        for elem in data:
            if not first_elem:
                position = np.argwhere(counts[:, 0] == elem.numpy())
                if position.shape == (1, 1):
                    count = counts[position[0, 0], 1]
                    count = int(count) + 1
                    counts[position[0, 0], 1] = count
                else:
                    counts = np.concatenate((counts, [[elem.numpy(), 1]]), axis=0)
            else:
                counts = np.array([[elem.numpy(), 1]])
                first_elem = False

        # sorting by the counts
        sort_indices = np.argsort(counts[:, 1])
        # inverting to have the most common on top
        sort_indices = np.flip(sort_indices)
        words = counts[:, 0]
        words_sorted = words[sort_indices]
        # only taking the 10000 most common words
        self.words_sorted_shortened = words_sorted[:self.number_words]
        print(self.words_sorted_shortened[:6])
        # print("finished counting")

    def prepare_data(self, text):
        '''
            Prepares a tf dataset by converting categorical features into one-hot features
            Caches, shuffles, batches and prefetches data

            Args:
                text: string
                training: bool

            Returns:
                data: the prepared tf dataset
        '''
        data = self.create_data(text)

        # create a bool.map if a word is one of the important words
        word_important = np.array([False] * data.shape[0])
        for i, elem in enumerate(data):
            word_important[i] = np.isin(elem.numpy(), self.words_sorted_shortened)

        # print("finished Bool-map")

        # create array of tuples of important elements in context-window of size 4
        dataset_array = np.zeros((np.sum(word_important) * 4, 2))
        elem_minus_one = 0
        elem_minus_two = 0
        elem_minus_three = 0
        elem_minus_four = 0
        j = 0
        for i, elem in enumerate(data):
            current_index = np.where(elem.numpy() == self.words_sorted_shortened)[0]
            if i >= 2 and word_important[i-2]:
                if word_important[i]:
                    dataset_array[j,0] = elem_minus_two
                    dataset_array[j,1] = current_index
                    j += 1
                if i >= 1 and word_important[i-1]:
                    dataset_array[j,0] = elem_minus_two
                    dataset_array[j,1] = elem_minus_one
                    j += 1
                if i >= 3 and word_important[i-3]:
                    dataset_array[j,0] = elem_minus_two
                    dataset_array[j,1] = elem_minus_three
                    j += 1
                if i >= 4 and word_important[i-4]:
                    dataset_array[j,0] = elem_minus_two
                    dataset_array[j,1] = elem_minus_four
                    j += 1
            elem_minus_four = elem_minus_three
            elem_minus_three = elem_minus_two
            elem_minus_two = elem_minus_one
            elem_minus_one = current_index

        print(f"Created array, dataset_size = {dataset_array.shape}")
        print(dataset_array)

        # create dataset from array
        dataset = tf.data.Dataset.from_tensor_slices(dataset_array)

        # change the dataset entries to int
        dataset = dataset.map(lambda element: (tf.cast(element[0], tf.int32), tf.cast(element[0], tf.int32)))

        # create one-hot encodings
        dataset = dataset.map(lambda word, target: (tf.one_hot(word, depth=self.number_words), tf.one_hot(target, depth=self.number_words)))

        # cache
        dataset = dataset.cache()

        # shuffle
        dataset = dataset.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

        # batch
        dataset = dataset.batch(64)

        # prefetch
        dataset = dataset.prefetch(20)

        return dataset

    def prepare_data_testing(self, text):
        '''
            Prepares a tf dataset by converting categorical features into one-hot features
            Caches, shuffles, batches and prefetches data

            Args:
                text: string
                training: bool

            Returns:
                data: the prepared tf dataset
        '''
        data = self.create_data(text)

        # create a bool.map if a word is one of the important words
        word_important = np.array([False] * data.shape[0])
        for i, elem in enumerate(data):
            word_important[i] = np.isin(elem.numpy(), self.words_sorted_shortened)

        # print("finished Bool-map")

        # create array of tuples of important elements in context-window of size 4
        dataset_array = np.zeros(np.sum(word_important))
        j = 0
        for i, elem in enumerate(data):
            current_index = np.where(elem.numpy() == self.words_sorted_shortened)[0]
            if word_important[i]:
                dataset_array[i] = current_index

        print(f"Created array, dataset_size = {dataset_array.shape}")
        print(dataset_array)

        # create dataset from array
        dataset = tf.data.Dataset.from_tensor_slices(dataset_array)

        # change the dataset entries to int
        dataset = dataset.map(lambda element: tf.cast(element, tf.int32))

        # create one-hot encodings
        dataset = dataset.map(lambda word: tf.one_hot(word, depth=self.number_words))

        # cache
        dataset = dataset.cache()

        # prefetch
        dataset = dataset.prefetch(20)

        return dataset

    def create_data(self, text):
        text = text.lower()

        # creating word tokens
        tokenizer = tf_txt.UnicodeScriptTokenizer()
        data = tokenizer.tokenize(text)

        # removing new line, digits, dots, commas, colons
        mask_to_be_excluded = tf.strings.regex_full_match(data, ".*(\.+|[\d]+|\n+|,+|:+).*")
        mask_to_be_excluded = tf.logical_not(mask_to_be_excluded)
        data = tf.boolean_mask(data, mask_to_be_excluded)

        return data
