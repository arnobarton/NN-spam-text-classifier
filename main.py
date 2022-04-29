import tensorflow as tf
import pandas as pd
from keras.preprocessing import sequence
from tensorflow import keras
!pip install tensorflow-datasets
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt

# get data files
!wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv
!wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"

df_train = pd.read_table(train_file_path, header=0, names=['indicates', 'text'], usecols=['indicates', 'text'])
df_test = pd.read_table(test_file_path, header=0, names=['indicates', 'text'], usecols=['indicates', 'text'])

df_train['indicates'] = df_train['indicates'].replace("ham", 0)
df_train['indicates'] = df_train['indicates'].replace("spam", 1)
df_test['indicates'] = df_test['indicates'].replace("ham", 0)
df_test['indicates'] = df_test['indicates'].replace("spam", 1)

train_data = tf.data.Dataset.from_tensor_slices((df_train['text'].values, df_train['indicates'].values))
test_data = tf.data.Dataset.from_tensor_slices((df_test['text'].values, df_test['indicates'].values))

tokenizer = tfds.deprecated.text.Tokenizer()

# To create vocabulary list from all data
vocabulary_set = set()

for text_tensor, _ in train_data.concatenate(test_data):
  some_tokens = tokenizer.tokenize(text_tensor.numpy())
  vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
encoder = tfds.deprecated.text.TokenTextEncoder(vocabulary_set)

def encode(text_tensor, label):
  encoded_text = encoder.encode(text_tensor.numpy())
  return encoded_text, label

def encode_map_fn(text, label):
  encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))

  encoded_text.set_shape([None])
  label.set_shape([])

  return encoded_text, label

train_data_encoded = train_data.map(encode_map_fn)
test_data_encoded = test_data.map(encode_map_fn)

BUFFER_SIZE = 1000
BATCH_SIZE = 32
train_dataset = (train_data_encoded.shuffle(BUFFER_SIZE).padded_batch(BATCH_SIZE))
test_dataset = (test_data_encoded.padded_batch(BATCH_SIZE))
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 32),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])

history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30)
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# function to predict messages based on model
# (should return list containing prediction and label, ex. [0.008318834938108921, 'ham'])
def predict_message(pred_text):
    pred_text = encoder.encode(pred_text)
    pred_text = tf.cast(pred_text, tf.float32)
    prediction = model.predict(tf.expand_dims(pred_text, 0)).tolist()
    if prediction[0][0] < 0.5:
      prediction = (' this is not spam')
    else:
      prediction = ('this is spam')
    return (prediction)
print("write your text message")
text = input()
pred_text = text
prediction = predict_message(pred_text)

  
print(prediction)
