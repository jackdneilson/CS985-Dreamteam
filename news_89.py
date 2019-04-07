from sklearn.datasets import fetch_20newsgroups
import tensorflow as tf
import numpy as np
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string

nltk.download('punkt')
nltk.download('stopwords')

news = fetch_20newsgroups(subset='all')

categories = range(0, 20)

N = round(len(news.data) * 0.7)
N_V = round(len(news.data) * 0.8)


x_train, y_train, x_val, y_val, x_test, y_test = news.data[0:N], news.target[0:N], \
                                                 news.data[N:N_V], news.target[N:N_V],\
                                                 news.data[N_V:], news.target[N_V:]

y_train = tf.keras.utils.to_categorical(y_train, len(categories))  # softmax reshape
y_test = tf.keras.utils.to_categorical(y_test, len(categories))
y_val = tf.keras.utils.to_categorical(y_val, len(categories))




from sklearn.feature_extraction.text import TfidfVectorizer
# vec = TfidfVectorizer(binary=False, strip_accents='ascii', use_idf=True,
#                       max_features=12500, stop_words='english')   # Turning to term frequency inverse doc freq
vec = TfidfVectorizer(binary=False, use_idf=True,
                      max_features=12500, stop_words='english')
x_train_tfidf = vec.fit_transform(x_train)
x_test_tfidf = vec.transform(x_test)
x_val_tfidf = vec.transform(x_val)

from sklearn.preprocessing import normalize  # Normalising the x train and test, seems to reduce performance
x_train_tfidf = normalize(x_train_tfidf, norm='l2', axis=1)
x_test_tfidf = normalize(x_test_tfidf, norm='l2', axis=1)
x_val_tfidf = normalize(x_val_tfidf, norm='l2', axis=1)



# Parameters
learning_rate = 0.001
num_steps = 16
batch_size = 64
display_step = 1
keep_prob = 0.5

# Network Parameters
n_hidden_1 = 512  # 1st layer number of neurons
n_hidden_2 = 512  # 2nd layer number of neurons
n_hidden_3 = 256
num_input = x_train_tfidf.shape[1]  # for each word in our dictionary
num_classes = len(categories)  # each category (20 in full model)

# tf Graph input
X = tf.placeholder("float", [None, num_input])  # x params stored here; each x is a tfidf for a news instance
Y = tf.placeholder("float", [None, num_classes])  # labels here

# weights and biases for each layer
# weights = {
#     'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
#     'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
#     'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
#     'out': tf.Variable(tf.random_normal([n_hidden_1, num_classes]))
# }

initialiser = tf.contrib.layers.xavier_initializer()

weights = {
    'h1': tf.Variable(initialiser([num_input, n_hidden_1])),
    'h2': tf.Variable(initialiser([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(initialiser([n_hidden_2, n_hidden_3])),
    'out': tf.Variable(initialiser([n_hidden_2, num_classes]))
}

biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'b3': tf.Variable(tf.random_normal([n_hidden_3])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def neural_net(x):
    # Hidden fully connected layer with n neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.layers.batch_normalization(layer_1)  # batch normalising

    layer_1 = tf.layers.dropout(layer_1, keep_prob)

    # Hidden fully connected layer
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])

    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.layers.batch_normalization(layer_2)
    layer_2 = tf.nn.dropout(layer_2, keep_prob)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.relu(layer_3)
    layer_3 = tf.layers.batch_normalization(layer_3)
    layer_3 = tf.nn.dropout(layer_3, keep_prob)

    # Output fully connected layer with a neuron for each class activated by softmax outside fn
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

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)

    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

    train_acc = list()
    val_accs = list()

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
              str(np.mean(train_acc[-batch_size:])))

        """ if validation accuracy some % off the train, do not go to next epoch"""
        """ also save checkpoints here... """


        for index, offset in enumerate(range(0, x_val_tfidf.shape[0], batch_size)):
            batch_x = x_val_tfidf[offset: offset + batch_size, ]
            batch_x = batch_x.todense()
            batch_x = np.asarray(batch_x)

            batch_y = y_val[offset: offset + batch_size, ]

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch " + str(index) + " Loss= " + \
                  "{:.4f}".format(loss) + ", Validation Accuracy= " + \
                  "{:.3f}".format(acc))
            val_accs.append(acc)
        print("Step Completed..." + "Mean Accuracy of Last Epoch:" + \
              str(np.mean(train_acc[-batch_size:])) + " Validation Accuracy: " + str(np.mean(val_accs[-3:])))

        # if np.mean(val_accs) <= (np.mean(train_acc[-batch_size:]) * 0.8) and step >= 1:
        #     print("Validation Accuracy too low, terminating...")
        #     break

        n_vals = round(x_val_tfidf.shape[0] / batch_size)

        if step >= 2 and np.mean(val_accs[-n_vals:]) < (np.mean(val_accs[-2*n_vals:-n_vals])):
            print("No Validation Improvement")
            break

        print("Would you like to continue training?....")
        answer = input('y/n')
        if answer == 'n':
            break

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
