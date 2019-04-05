from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

tf.logging.set_verbosity(tf.logging.INFO)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()  # split/load

print("x_train shape::", x_train.shape, "y_train.shape::", y_train.shape)  # looking at shapes


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

x_train = x_train.astype('float32') / 255  # Normalising
x_test = x_test.astype('float32') / 255

#tf.cast(x_train, "float32")


x_val = x_train[48000:]
y_val = y_train[48000:]  # validation sets, 0.2 of train

x_train = x_train[0:48000]  # no overlap
y_train = y_train[0:48000]


input_shape = (1, 28, 28)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
X_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 1)

y_train = tf.keras.utils.to_categorical(y_train, 10)

#y_train = tf.reshape(y_train, [48000, 10])  # for softmax

y_test = tf.keras.utils.to_categorical(y_test, 10)

#y_test = tf.reshape(y_test, [len(y_test), 10])

y_val = tf.keras.utils.to_categorical(y_val, 10)

#y_val = tf.reshape(y_val, [len(y_val), 10])



model = tf.keras.Sequential()

conv1 = model.add(tf.layers.Conv2D(filters=32, kernel_size=[3, 3], padding="same", input_shape=(28,28,1),
                                   activation=tf.nn.relu))  # convolutional layer

# pool1 = model.add(tf.layers.MaxPooling2D(inputs=conv1, pool_size=[2, 2], strides=1))  # pooling layer
# drop1 = model.add(tf.layers.Dropout(0.25, inputs=pool1))  # for simplicity
#
# conv2 = model.add(tf.layers.Conv2D(inputs=drop1, filters=32, kernel_size=2, padding='same', activation='relu'))
# pool2 = model.add(tf.layers.MaxPooling2D(inputs=conv2, pool_size=2))
# drop2 = model.add(tf.layers.Dropout(0.25, inputs=pool2))
#
# flat = model.add(tf.layers.Flatten(inputs=drop2))
# dense = model.add(tf.layers.Dense(128, inputs=flat, activation = tf.nn.relu))
# soft = model.add(tf.layers.Dense(10, inputs=dense, activation='softmax'))
#


model.add(tf.layers.MaxPooling2D(pool_size=(2, 2), strides=1))
#randomly turn neurons on and off to improve convergence
model.add(tf.layers.Dropout(0.25))
#flatten since too many dimensions, we only want a classification output
model.add(tf.layers.Flatten())
#fully connected to get all relevant data
model.add(tf.layers.Dense(128, activation='relu'))
#one more dropout for convergence' sake :)
model.add(tf.layers.Dropout(0.5))
#output a softmax to squash the matrix into output probabilities
model.add(tf.layers.Dense(10, activation='softmax'))




model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

batch_size = 1
num_epoch = 1
model_log = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=num_epoch,
          verbose=1,
          validation_data=(x_val, y_val), steps_per_epoch=1)


model.summary()


score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
