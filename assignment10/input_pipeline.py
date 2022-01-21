'''
authors: tnortmann, hsanna, lmcdonald
'''

import tensorflow as tf
import tensorflow_text as tf_txt
import numpy as np

number_words = 10000

def prepare_data(text):
    '''
        Prepares a tf dataset by converting categorical features into one-hot features
        Caches, shuffles, batches and prefetches data

        Args:
            data: tf dataset

        Returns:
            data: the prepared tf dataset
    '''

    # make text to lower
    text = text.lower()

    # creating word tokens
    tokenizer = tf_txt.UnicodeScriptTokenizer()
    data = tokenizer.tokenize(text)

    # removing new line, digits, dots, commas, colons
    mask_to_be_excluded = tf.strings.regex_full_match(data, ".*(\.+|[\d]+|\n+|,+|:+).*")
    mask_to_be_excluded = tf.logical_not(mask_to_be_excluded)
    data = tf.boolean_mask(data, mask_to_be_excluded)
    data_len = data.shape[0]
    # data = tf.data.Dataset.from_tensor_slices(data)

    # getting the counts of the different words
    first_elem = True
    for elem in data:
        if not first_elem:
            position = np.argwhere(counts[:,0] == elem.numpy())
            if position.shape == (1,1):
                count = counts[position[0,0], 1]
                count = int(count) + 1
                counts[position[0,0], 1] = count
            else:
                counts = np.concatenate((counts, [[elem.numpy(), 1]]), axis = 0)
        else:
            counts = np.array([[elem.numpy(), 1]])
            first_elem = False

    # sorting by the counts
    sort_indices = np.argsort(counts[:,1])
    # inverting to have the most common on top
    sort_indices = np.flip(sort_indices)
    words = counts[:,0]
    words_sorted = words[sort_indices]
    # only taking the 10000 most common words
    words_sorted_shortened = words_sorted[:number_words]

    print("finished counting")

    word_important = np.array([False] * data.shape[0])
    for i, elem in enumerate(data):
        word_important[i] = np.isin(elem.numpy(), words_sorted_shortened)

    print("finished Bool-map")

    dataset_array = np.array([("","")] * np.sum(word_important) * 4)
    elem_minus_one = ""
    elem_minus_two = ""
    elem_minus_three = ""
    elem_minus_four = ""
    j = 0
    for i, elem in enumerate(data):
        if i >= 2 and word_important[i-2]:
            if word_important[i]:
                dataset_array[j] = (elem_minus_two, elem.numpy())
                j += 1
            if i >= 1 and word_important[i-1]:
                dataset_array[j] = (elem_minus_two, elem_minus_one)
                j += 1
            if i >= 3 and word_important[i-3]:
                dataset_array[j] = (elem_minus_two, elem_minus_three)
                j += 1
            if i >= 4 and word_important[i-4]:
                dataset_array[j] = (elem_minus_two, elem_minus_four)
                j += 1
        elem_minus_four = elem_minus_three
        elem_minus_three = elem_minus_two
        elem_minus_two = elem_minus_one
        elem_minus_one = elem.numpy()

    print(f"Created array, dataset_size = {dataset_array.shape}")

    dataset = tf.data.Dataset.from_tensor_slices(dataset_array)
    #
    # dataset_array = np.array([])
    # for i in range(data_len - 5):
    #     words_tensors = data.take(5)
    #     words = np.array([])
    #     for elem in words_tensors:
    #         words = np.append(words, elem.numpy())
    #     if np.isin(words[2], words_sorted_shortened):
    #         for j in [0,1,3,4]:
    #             if np.isin(words[j], words_sorted_shortened):
    #                 np.append(dataset_array, (words[2],words[j]))
    #     data.skip(1)
    # dataset = tf.data.Dataset.from_tensor_slices(dataset_array)

   # dataset = dataset.map(lambda element: (tf.one_hot(element[0], depth=number_words), tf.one_hot(element[1], depth=number_words)))

    # cache
    dataset = dataset.cache()

    # shuffle
    dataset = dataset.shuffle(1000, reshuffle_each_iteration = True, seed = 42)

    # batch
    dataset = dataset.batch(64)

    # prefetch
    dataset = dataset.prefetch(20)

    return dataset