from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']


news_train = fetch_20newsgroups(subset='train', categories=categories)  # getting data
news_test = fetch_20newsgroups(subset='test', categories=categories)


y_train, y_test = news_train.target, news_test.target  # labels


from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(binary=True, use_idf=True)  # Turning to term frequency inverse doc freq
x_train_tfidf = vec.fit_transform(news_train.data)
x_test_tfidf = vec.transform(news_test.data)

# x_train_tfidf = np.asarray(x_train_tfidf) # numpy array, not sure
# x_test_tfidf = np.asarray(x_test_tfidf)


def convert_sparse_matrix_to_sparse_tensor(X):
    # sparse matrix can't be input to tensors I think?
    coo = X.tocoo()
    indices = np.mat([coo.row, coo.col]).transpose()
    output = tf.SparseTensorValue(indices, coo.data, coo.shape)
    return (output)


x_train_tfidf = convert_sparse_matrix_to_sparse_tensor(x_train_tfidf)
x_test_tfidf = convert_sparse_matrix_to_sparse_tensor(x_test_tfidf)

# Parameters
learning_rate = 0.1
num_steps = 10
batch_size = 64
display_step = 100

# Network Parameters
n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_input = 35788 # for each word in our dictionary
num_classes = len(categories)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

#
# x_train_tfidf = tf.sparse.to_dense(x_train_tfidf)
# x_test_tfidf = tf.sparse.to_dense(x_test_tfidf)
#


# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):

        for index, offset in enumerate(range(0, 35788, batch_size)):
            sess.run(init)
            # batch_x = x_train_tfidf.values[offset: offset + batch_size]

            batch_x = x_train_tfidf[offset: offset + batch_size]  # HERE is the problem, can't batch up data


            #batch_x = np.asarray(batch_x)
            # batch_x = tf.sparse.to_dense(batch_x)
            batch_y = y_train[offset: offset + batch_size]
        # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))

    print("Optimization Finished!")

    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: x_test_tfidf,
                                        Y: y_test}))
