import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.datasets import fetch_20newsgroups
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Embedding, GlobalMaxPooling1D, MaxPooling1D, Dropout, SpatialDropout1D, AveragePooling1D, GlobalAveragePooling1D
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.regularizers import l2

# Clear session (helps with some OOM errors)
tf.keras.backend.clear_session()

plt.style.use('ggplot')


def plot_history(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()


tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 0.00001
MAX_LENGTH = 200
NO_EPOCHS = 10
BATCH_SIZE = 128

# Fetch the dataset
news_train = fetch_20newsgroups(subset='train', shuffle=True)
news_test = fetch_20newsgroups(subset='test', shuffle=True)

categories = range(0, 20)

# Tokenize the input to be sequences of words, then pad to a max length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_train.data)
x_train = tokenizer.texts_to_sequences(news_train.data)
x_train = pad_sequences(x_train, maxlen=MAX_LENGTH, padding='post')
x_test = tokenizer.texts_to_sequences(news_test.data)
x_test = pad_sequences(x_test, maxlen=MAX_LENGTH, padding='post')
y_train, y_test = news_train.target, news_test.target
y_train = to_categorical(y_train, len(categories))
y_test = to_categorical(y_test, len(categories))
vocab_size = len(tokenizer.word_index) + 1


# Define the Convolutional Neural Network model
model = Sequential()
model.add(Embedding(vocab_size, 64))
model.add(SpatialDropout1D(0.5))
model.add(Conv1D(32, activation='relu', kernel_size=6, kernel_regularizer=l2(0.0001)))
model.add(GlobalAveragePooling1D())
# model.add(SpatialDropout1D(0.5))
# model.add(Conv1D(64, activation='relu', kernel_size=8, kernel_regularizer=l2(0.0001)))
# model.add(GlobalMaxPooling1D())
# model.add(Dropout(0.5))
# model.add(Dense(30, activation='relu', kernel_regularizer=l2(0.00001)))
# model.add(Dropout(0.5))
# model.add(Dense(30, activation='relu', kernel_regularizer=l2(0.00001)))
model.add(Dense(20, name='output', activation='softmax'))

# Define the optimizer for the model
sgd_opt = SGD(lr=LEARNING_RATE)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['acc']
)

print(model.summary())

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=NO_EPOCHS, batch_size=BATCH_SIZE)
plot_history(history)
