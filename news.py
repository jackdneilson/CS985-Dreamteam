from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

categories = ['alt.atheism', 'soc.religion.christian','comp.graphics', 'sci.med'] # subcat for quick build


news_train = fetch_20newsgroups(subset='train', categories=categories)  # getting data
news_test = fetch_20newsgroups(subset='test', categories=categories)


y_train, y_test = news_train.target, news_test.target  # labels


y_train = tf.keras.utils.to_categorical(y_train, len(categories)) # not sure if want softmax here?
y_test = tf.keras.utils.to_categorical(y_test, len(categories))


# y_train = tf.reshape(y_train, [len(news_train.data), len(categories)])
# y_test = tf.reshape(y_test, [len(news_test.data), len(categories)])



from sklearn.feature_extraction.text import TfidfVectorizer
vec = TfidfVectorizer(binary=True, use_idf=True)  # Turning to term frequency inverse doc freq
x_train_tfidf = vec.fit_transform(news_train.data)
x_test_tfidf = vec.transform(news_test.data)


from sklearn.preprocessing import normalize  # Normalising the x train and test
x_train_tfidf = normalize(x_train_tfidf, norm='l1', axis=1)
x_test_tfidf = normalize(x_test_tfidf, norm='l1', axis=1)


# Parameters
learning_rate = 0.001
num_steps = 1
batch_size = 128
display_step = 100
keep_prob = 0.5

# Network Parameters
n_hidden_1 = 512 # 1st layer number of neurons
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
# def neural_net(x):
#     # Hidden fully connected layer with 256 neurons
#     layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
#     # Hidden fully connected layer with 256 neurons
#     layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
#     # Output fully connected layer with a neuron for each class
#     out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
#
#     # out_layer = tf.clip_by_value(out_layer, clip_value_min=0.00001, clip_value_max=1)
#
#     return out_layer


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    # We first get weights and biases, then batch normalise, then dropout (or vice versa) and activate last
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1_b = tf.layers.batch_normalization(layer_1) # batch normalising
    layer_1_d = tf.nn.dropout(layer_1_b, keep_prob) # dropout
    layer_1_f = tf.nn.relu(layer_1_d) # activation with relu

    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1_d, weights['h2']), biases['b2'])
    layer_2_b = tf.layers.batch_normalization(layer_2)
    layer_2_d = tf.nn.dropout(layer_2_b, keep_prob)
    layer_2_f = tf.nn.relu(layer_2_d)

    # Output fully connected layer with a neuron for each class
    out_layer = tf.nn.relu(tf.matmul(layer_2_d, weights['out']) + biases['out'])
    out_layer_b = tf.layers.batch_normalization(out_layer)
    out_layer_d = tf.nn.dropout(out_layer, keep_prob)
    out_layer_f = tf.nn.relu(out_layer_d)

    # out_layer_f = tf.clip_by_value(out_layer, clip_value_min=0.00001, clip_value_max=1)

    return out_layer


# Construct model
logits = neural_net(X)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))

# loss_op = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(
#     logits=logits, labels=Y))
#
# loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))
#


# def loss_op_fix(logits, labels):
#
#     logits = tf.clip_by_value(logits, clip_value_min=0.00001, clip_value_max=1)
#
#     loss_op = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(
#         logits=logits, labels=labels))
#
#     # w = tf.where(tf.is_nan(X), tf.ones_like(X) * 0, X);  # if w is nan use 1 * NUMBER else use element in w
#     #
#     # for i in loss_op:
#     #     if np.isnan(i) is True:
#     #         i = 0
#     return(loss_op)
#
# loss_op = loss_op_fix(logits, Y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()



# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    for step in range(1, num_steps + 1):

        for index, offset in enumerate(range(0, 35788, batch_size)):

            batch_x = x_train_tfidf[offset: offset + batch_size, ]
            batch_x = batch_x.todense()
            batch_x = np.asarray(batch_x)

            batch_y = y_train[offset: offset + batch_size]


        # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})

            if step % display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                     Y: batch_y})
                print("Step " + str(step) + ", Minibatch " + str(index) + " Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        print("Step Completed...")

    print("Optimization Finished!")


    ### NOW THE TEST SET NEEDS TO BE BATCHED UP!

    print("Testing Accuracy:", \
          sess.run(accuracy, feed_dict={X: x_test_tfidf,
                                        Y: y_test}))
