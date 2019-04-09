import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Embedding, SpatialDropout1D, GlobalAveragePooling1D
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.regularizers import l2

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
NO_EPOCHS = 8
BATCH_SIZE = 128
SEED = 12345

# Fetch the dataset and split into training, validation and testing sets.
news = fetch_20newsgroups(subset='all')
train_index_end = round(0.7 * len(news.data))
val_index_end = round(0.8 * len(news.data))
news_train, news_val, news_test = news.data[:train_index_end],\
                                  news.data[train_index_end:val_index_end],\
                                  news.data[val_index_end:]

x_train, y_train = news.data[:train_index_end], news.target[:train_index_end]
x_val, y_val = news.data[train_index_end:val_index_end], news.target[train_index_end:val_index_end]
x_test, y_test = news.data[val_index_end:], news.target[val_index_end:]

categories = range(0, 20)

# Tokenize the input to be sequences of words, then pad to a max length
tokenizer = Tokenizer()
tokenizer.fit_on_texts(news_train)
x_train = pad_sequences(tokenizer.texts_to_sequences(x_train), maxlen=MAX_LENGTH, padding='post')
x_val = pad_sequences(tokenizer.texts_to_sequences(x_val), maxlen=MAX_LENGTH, padding='post')
x_test = pad_sequences(tokenizer.texts_to_sequences(x_test), maxlen=MAX_LENGTH, padding='post')
y_train = to_categorical(y_train, len(categories))
y_val = to_categorical(y_val, len(categories))
y_test = to_categorical(y_test, len(categories))
vocab_size = len(tokenizer.word_index) + 1


# Define the Convolutional Neural Network model
def create_model(learning_rate=0.00001, feature_maps=32, kernel_size=6, weight_decay_rate=0.0001):
    model = Sequential()
    model.add(Embedding(vocab_size, 128))
    model.add(SpatialDropout1D(0.2))
    model.add(Conv1D(feature_maps, activation='relu', kernel_size=kernel_size, kernel_regularizer=l2(weight_decay_rate)))
    model.add(GlobalAveragePooling1D())
    model.add(Dense(20, name='output', activation='softmax'))

    opt = Adam(lr=learning_rate)

    # Compile and return the model
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )
    return model


# Grid search for the best combinations of hyperparameters
learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
epochs = range(12)
batches = [32, 64, 128, 256]
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=256, verbose=1)
param_grid = dict(learning_rate=learning_rates)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)

x_search = pad_sequences(tokenizer.texts_to_sequences(news.data), maxlen=MAX_LENGTH, padding='post')
y_search = to_categorical(news.target, len(categories))
grid_result = grid.fit(x_search, y_search)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Define callback functions to save a checkpoint after training as well as tensorboard logs
checkpoint_path = "news-{}-{}-{}-{}-{}.ckpt".format(2, LEARNING_RATE, NO_EPOCHS, BATCH_SIZE, SEED)
cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=0, period=1)
now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
tensorboard = tf.keras.callbacks.TensorBoard(log_dir="./logs/run-{}".format(now), write_graph=True, update_freq="batch")

# model = create_model()
# history = model.fit(x_train,
#                     y_train,
#                     validation_data=(x_val, y_val),
#                     epochs=NO_EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[cp_callback, tensorboard])
# loss, accuracy = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE, verbose=0)
# plot_history(history)
# print("Test Loss: " + str(loss))
# print("Test Accuracy: " + str(accuracy))
