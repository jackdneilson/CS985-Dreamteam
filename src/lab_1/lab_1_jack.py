import tensorflow as tf
# from sklearn.datasets import fetch_california_housing
#
# housing = fetch_california_housing()

x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x * x * y + y + 2

