from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med']


news_train = fetch_20newsgroups(subset='train', shuffle=True)  # getting data
news_test = fetch_20newsgroups(subset='test', shuffle=True)


categories = range(0, 20)

y_train, y_test = news_train.target, news_test.target  # labels

y_train = tf.keras.utils.to_categorical(y_train, len(categories)) # softmax reshape
y_test = tf.keras.utils.to_categorical(y_test, len(categories))




from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(binary=True, use_idf=True)  # Turning to term frequency inverse doc freq
x_train_tfidf = vec.fit_transform(news_train.data)
x_test_tfidf = vec.transform(news_test.data)


from sklearn.preprocessing import normalize  # Normalising the x train and test, seems to reduce performance
x_train_tfidf = normalize(x_train_tfidf, norm='l2', axis=1)
x_test_tfidf = normalize(x_test_tfidf, norm='l2', axis=1)


# Parameters
learning_rate = 0.03
num_steps = 1
batch_size = 256
display_step = 1
keep_prob = 0.5

# Network Parameters
n_hidden_1 = 512 # 1st layer number of neurons
n_hidden_2 = 256  # 2nd layer number of neurons
n_hidden_3 = 128
num_input = x_train_tfidf.shape[1]  # for each word in our dictionary
num_classes = len(categories)  # each category (20 in full model)

# tf Graph input
X = tf.placeholder("float", [None, num_input])  # x params stored here; each x is a tfidf for a news instance
Y = tf.placeholder("float", [None, num_classes])  # labels here

# weights and biases for each layer
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


# def neural_net(x):
#     # Hidden fully connected layer with n neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising
#     # layer_1 = tf.nn.dropout(layer_1, keep_prob)
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising
#
#     layer_1 = tf.nn.tanh(layer_1)  # two at same time, or either; this seems a little bit better
#
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # layer_2 = tf.layers.batch_normalization(layer_2)
#     # layer_2 = tf.nn.dropout(layer_2, keep_prob)
#     # layer_2 = tf.nn.relu(layer_2)
#     # layer_2 = tf.layers.batch_normalization(layer_2)
#     # layer_2 = tf.nn.dropout(layer_2, 0.75)
#
#     layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
#     # layer_3 = tf.nn.dropout(layer_3, keep_prob)
#     # layer_3 = tf.nn.relu(layer_3)
#     # layer_3 = tf.layers.batch_normalization(layer_3)
#
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_3, weights['out']) + biases['out']
#     # out_layer = tf.nn.relu(out_layer)
#
#     # out_layer = tf.nn.tanh(out_layer) # is activated with softmax outside of fn
#     return out_layer

def neural_net(x):
    # Hidden fully connected layer with n neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising
    # layer_1 = tf.nn.dropout(layer_1, keep_prob)
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising

    # layer_1 = tf.nn.tanh(layer_1)  # two at same time, or either; this seems a little bit better

    # layer_1 = tf.nn.dropout(layer_1, keep_prob=0.9)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    # out_layer = tf.nn.relu(out_layer)

    # out_layer = tf.nn.tanh(out_layer) # is activated with softmax outside of fn
    return out_layer



# def neural_net(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising
#     # layer_1 = tf.nn.dropout(layer_1, 0.9)
#     layer_1 = tf.nn.relu(layer_1)
#     layer_1 = tf.layers.batch_normalization(layer_1) # batch normalising
#
#     layer_1 = tf.nn.tanh(layer_1)  # two at same time, or either; this seems a little bit better
#
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # layer_2 = tf.layers.batch_normalization(layer_2)
#     layer_2 = tf.nn.relu(layer_2)
#     layer_2 = tf.layers.batch_normalization(layer_2)
#
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#     # out_layer = tf.nn.relu(out_layer)
#
#     # out_layer = tf.nn.tanh(out_layer) # is activated with softmax outside of fn
#     return out_layer

# def neural_net(x):
#     # Hidden fully connected layer with n neurons
#     # We first get weights and biases, then batch normalise, then dropout (or vice versa) and activate last
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     layer_1_b = tf.layers.batch_normalization(layer_1) # batch normalising
#     layer_1_d = tf.nn.dropout(layer_1_b, keep_prob) # dropout
#     layer_1_f = tf.nn.relu(layer_1_d) # activation with relu
#
#     # Hidden fully connected layer with n neurons
#     layer_2 = tf.add(tf.matmul(layer_1_f, weights['h2']), biases['b2'])
#     layer_2_b = tf.layers.batch_normalization(layer_2)
#     layer_2_d = tf.nn.dropout(layer_2_b, keep_prob)
#     layer_2_f = tf.nn.relu(layer_2_d)
#
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2_f, weights['out']) + biases['out']
#     out_layer_b = tf.layers.batch_normalization(out_layer)
#     # out_layer_f = tf.nn.relu(out_layer_b)
#
#     return out_layer_b



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


with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    train_acc = list()

    for step in range(1, num_steps + 1):

        for index, offset in enumerate(range(0, x_train_tfidf.shape[0], batch_size)):

            batch_x = x_train_tfidf[offset: offset + batch_size, ]
            batch_x = batch_x.todense()
            batch_x = np.asarray(batch_x)

            batch_y = y_train[offset: offset + batch_size, ]


        # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0 or step == 1 or True:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch " + str(index) + " Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
                train_acc.append(acc)
        print("Step Completed..." + "Mean Accuracy of Last Epoch:" + \
              str(np.mean(train_acc[-36:])))

    print("Optimization Finished!")


    ### NOW THE TEST SET NEEDS TO BE BATCHED UP!

    test_accs = list()

    for index, offset in enumerate(range(0, x_test_tfidf.shape[0], batch_size)):


        batch_x = x_test_tfidf[offset: offset + batch_size, ]
        batch_x = batch_x.todense()
        batch_x = np.asarray(batch_x)

        batch_y = y_test[offset: offset + batch_size, ]

        # Run optimization -- this continues training with the test set, which maybe we shouldn't do?
        # sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

        # Calculate batch loss and accuracy
        loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                             Y: batch_y})
        print("Minibatch " + str(index) + " Loss= " + \
              "{:.4f}".format(loss) + ", Testing Accuracy= " + \
              "{:.3f}".format(acc))

        test_accs.append(acc)

    print("Testing Accuracy:", np.mean(test_accs))

    saver = tf.train.Saver()

    save_path = saver.save(sess, "/tmp/model.ckpt")
    print("SAVED MODEL TO PATH: %s" % save_path)


import matplotlib.pyplot as plt

N = round(x_train_tfidf.shape[0] / batch_size)

## plotting by epoch
# plt.ylim(0.2, 1.1)
# plt.xlim(0, 110)
# plt.title("Accuracy History for each Batch and Epoch")
# e1, = plt.plot(range(0, N), train_acc[0:N], color="r")
# e2, = plt.plot(range(N, 2*N), train_acc[N:(2*N)], color="b")
# e3, = plt.plot(range(2*N, len(train_acc)), train_acc[71:], color="g")
# plt.legend((e1, e2, e3), ('Epoch 1', 'Epoch 2', 'Epoch 3'), loc="right")
# plt.show()


## plotting by epoch
plt.ylim(0.2, 1.1)
plt.xlim(0, 110)
plt.title("Accuracy History for each Batch and Epoch")
e1, = plt.plot(range(0, N), train_acc[0:N], color="r")
e2, = plt.plot(range(N-1, (2*N)), train_acc[N-1:(2*N)], color="b")
# e3, = plt.plot(range(2*N, len(train_acc)), train_acc[71:], color="g")
plt.legend((e1, e2), ('Epoch 1', 'Epoch 2'), loc="right")
plt.show()



plt.plot(train_acc)
plt.show()

plt.plot(test_accs)
plt.show()


#
# tf.math.confusion_matrix(labels= y_test, predictions=prediction, num_classes=4)
# cm = tf.confusion_matrix(labels=tf.argmax(batch_y, 1), predictions=tf.argmax(batch_x, 1))
