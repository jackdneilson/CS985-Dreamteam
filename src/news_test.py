import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.utils import to_categorical

tf.logging.set_verbosity(tf.logging.INFO)

LEARNING_RATE = 0.01

# Fetch the dataset
news_train = fetch_20newsgroups(subset='train', shuffle=True)
news_test = fetch_20newsgroups(subset='test', shuffle=True)

categories = range(0, 20)

tfidf_vec = TfidfVectorizer(use_idf=True)
x_train_idf = tfidf_vec.fit_transform(news_train.data)
#x_test_idf = tfidf_vec.fit_transform(news_test.data)
y_train, y_test = news_train.target, news_test.target
y_train = to_categorical(y_train, len(categories))
y_test = to_categorical(y_test, len(categories))

print(x_train_idf.shape)
print(x_train_idf.shape[0])
print(y_train.shape)
print(y_train[0].shape)
print(len(tfidf_vec.vocabulary_))

# Define the Neural Network model
# int(1.5 * len(tfidf_vec.vocabulary_))
model = Sequential()
model.add(Dense(100, activation='relu', input_dim=len(tfidf_vec.vocabulary_)))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(20, name='output', activation='softmax'))

# Define the optimizer for the model
sgd_opt = SGD(lr=LEARNING_RATE)

# Compile the model
model.compile(
    optimizer='SGD',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(model.summary())

# The data is far too large to fit into memory, so define a generator to yield smaller chunks
def generator(features, labels, batch_size):
    for i in range(features.shape[0]):
        yield features[i:i + 1, ].todense(), labels[i:i + 1, ]


# Fit the model to the training data
model.fit_generator(
    generator(x_train_idf, y_train, 1),
    epochs=32,
    steps_per_epoch=x_train_idf.shape[0])


#model.fit(x_train_idf, y_train, epochs=1, verbose=1, batch_size=10)


