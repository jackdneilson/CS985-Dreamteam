import tensorflow as tf
import numpy as np
from sklearn.datasets import fetch_california_housing

N_EPOCHS = 1000
LEARNING_RATE = 0.01

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

x = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="x")
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
theta = tf.Variable(tf.random_uniform([n + 1, 1], -1.0, 1.0), name="theta")

y_pred = tf.matmul(x, theta, name="predictions")
error = y_pred - y
mse = tf.reduce_mean(tf.square(error), name="mse")
gradients = 2/m * tf.matmul(tf.transpose(x), error)
training_op = tf.assign(theta, theta - LEARNING_RATE * gradients)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(N_EPOCHS):
        if epoch % 100 == 0:
            print("Epoch", epoch, "MSE", mse.eval())
        sess.run(training_op)

    best_theta = theta.eval()
    print(best_theta)
#TODO: Ask - gives NaN, theta is variable not float.
