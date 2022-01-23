'''
authors: tnortmann, lmcdonald
'''

import tensorflow as tf
import numpy as np
import tensorflow.keras as K
import input_pipeline
import training_loop
from skip_gram import SkipGram
import re
import nltk
nltk.download("punkt")

# Hyperparameters
num_epochs = 10
alpha = 0.1
embedding_dimensions = 64
number_vocabulary = 10000

# pre-process text
def clean_text(text):
  text = text.lower()
  text = re.sub(r'\n', " ", text)
  text = re.sub(r'[^\w\s]', "", text)
  text = re.sub(r'\d+', "", text)
  return text

# tokenize
def tokenize(text):
  return nltk.word_tokenize(text)

# only use num most frequent words
def get_most_frequent(tokens, num):
  fdist = nltk.FreqDist(tokens)
  return fdist.most_common(num)

def create_vocab(most_frequent):
  vocab = {}
  for i, token in enumerate(most_frequent):
    vocab.update({token[0]:i})

  return vocab

def invert_vocab(vocab):
  inverse = {}

  for key, value in vocab.items():
    inverse.update({value:key})

  return inverse

def tokens_to_ints(tokens, vocab):

  int_tokens = []

  for token in tokens:
    if token in vocab:
      int_tokens.append(vocab.get(token))

  return int_tokens

# read in file
bible_text = ""
with open("dataset/bible.txt") as f:
  bible_text = f.read()

# preprocessing
bible_text_clean = clean_text(bible_text)
bible_tokens = tokenize(bible_text_clean)
bible_tokens_most_frequent = get_most_frequent(bible_tokens, num=number_vocabulary)
bible_vocab = create_vocab(bible_tokens_most_frequent)
bible_vocab_inverse = invert_vocab(bible_vocab)

bible_integer_tokens = tokens_to_ints(bible_tokens, bible_vocab)

# create dataset
def generator():

  for input, target in input_pipeline.skip_gram_generator(bible_integer_tokens, bible_vocab_inverse, window_size=4):
    yield input, target

data = tf.data.Dataset.from_generator(
    generator,
    output_signature=(
        tf.TensorSpec(shape=(), dtype=tf.int64),
        tf.TensorSpec(shape=(), dtype=tf.int64))
    )

train_data = data.apply(input_pipeline.prepare_data)

test_tokens = tokens_to_ints(["holy", "father", "wine", "poison", "love", "strong", "day"], bible_vocab)
test_data = tf.data.Dataset.from_tensor_slices(test_tokens)
test_data = test_data.apply(input_pipeline.prepare_data_test)

# Initialize Model
model = SkipGram(embedding_dimensions=embedding_dimensions, number_vocabulary=number_vocabulary)


# optimizer
optimizer = K.optimizers.SGD(alpha)

# # initialize lists for later visualization.
train_losses = []

# We train for num_epochs epochs.
for epoch in range(num_epochs):
    # training (and checking in with training)
    epoch_losses = []
    for input, target in train_data:
        train_loss = training_loop.train_step(model, input, target, optimizer, number_vocabulary=number_vocabulary)
        epoch_losses.append(train_loss)


    # track training loss
    train_losses.append(tf.reduce_mean(epoch_losses))
    print(f"epoch {epoch} finished with loss of {train_losses[-1]}")

    k = 5
    embeddings = model.embedding_matrix.numpy()
    cosine_loss = tf.keras.losses.CosineSimilarity(axis=0)
    best_match = ""
    for elem in test_data:
        embedding_elem = model(elem)
        cosine_losses = np.ones(embeddings.shape[0])
        for i in range(embeddings.shape[0]):
            embedding = tf.nn.embedding_lookup(embeddings, i)
            cosine_losses[i] = cosine_loss(embedding_elem, embedding)
        indices = np.argsort(cosine_losses)
        indices = indices[:k]
        best_matches = ""
        for j in range(k):
            best_matches += str(bible_vocab_inverse.get(indices[j])) + " "
        elem_string = bible_vocab_inverse.get(elem.numpy())
        print(f"The most similar elements to {elem_string} are {best_matches}")


