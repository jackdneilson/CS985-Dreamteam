from __future__ import absolute_import, division, print_function
import argparse
import tensorflow as tf
import numpy as np


##Logging
##Saving the model
tf.logging.set_verbosity(tf.logging.INFO)

def mlp_network(combination, learning_rate, epochs, batches, seed):
    tf.set_random_seed(seed)

    checkpoint_path = "mnist-{}-{}-{}-{}-{}.ckpt".format(combination,learning_rate,epochs,batches,seed)
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose=1, period=1)
    
    ((X_train, y_train), (X_test, y_test)) = tf.keras.datasets.mnist.load_data()

    X_train = X_train/np.float32(255)
    y_train = y_train.astype(np.int32)

    X_test = X_test/np.float32(255)
    y_test = y_test.astype(np.int32)
    
    if combination == 1 or combination == 3:
        model = dnn(combination)
    
    if combination == 2 or combination == 4:
        model, X_train, X_test = cnn(combination, X_train, X_test)
    
    model.summary()
    
    adam = tf.keras.optimizers.Adam(lr=learning_rate)
    
    model.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
             optimizer=adam,
             metrics=['accuracy'])
    
    model.fit(X_train, y_train, batch_size=batches,
              epochs=epochs, verbose=1, validation_split=0.1,
              shuffle="batch", callbacks = [cp_callback])
    
    score = model.evaluate(X_test, y_test, verbose=0)
    
    print("Test accuracy: ", score[1])
    
    print("Restoring a model")
    model1 = dnn(combination)
    model1.load_weights(checkpoint_path)
    model1.compile(loss=tf.keras.losses.sparse_categorical_crossentropy,
             optimizer=adam,
             metrics=['accuracy'])    
    loss1, acc1 = model1.evaluate(X_test, y_test)
    print("Restored model1, accuracy: {:5.2f}%".format(100*acc1))

    
    
def dnn(combination):
    model = tf.keras.Sequential()
    
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    model.add(tf.keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(256, activation=tf.nn.relu))
    if combination == 3:
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=0.8))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

    return model

def cnn(combination, X_train, X_test):
    if tf.keras.backend.image_data_format() == 'channels_first':
        X_train_reshaped = X_train.reshape(X_train.shape[0], 1, 28, 28)
        X_test_reshaped = X_test.reshape(X_test.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        X_train_reshaped = X_train.reshape(X_train.shape[0], 28, 28, 1)
        X_test_reshaped = X_test.reshape(X_test.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)
        
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3), 
                     activation=tf.nn.relu, 
                     input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) 
    model.add(tf.keras.layers.Conv2D(64, kernel_size=(3,3), 
                     activation=tf.nn.relu)) 
    model.add(tf.keras.layers.AveragePooling2D(pool_size=(2,2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    if combination == 4:
        model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dropout(rate=0.2))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))
    
    return model, X_train_reshaped, X_test_reshaped      
    
def check_param_is_float(param, value):

    try:
        value = float(value)
    except:
        print("{} must be float".format(param))
        quit(1)
    return value    

def check_param_is_int(param, value):
    
    try:
        value = int(value)
    except: 
        print("{} must be integer".format(param))
        quit(1)
    return value  

# 4 Combinations
# Combination 1: MLP with 2 hidden layers
# Combination 2: CNN with 1 hidden layer
# Combination 3: MLP with 3 hidden layers
# Combination 4: CNN with 2 hidden layers

#mlp_network(3,0.001,10,128,12345)

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Assignment Program")
    arg_parser.add_argument("combination", help="Flag to indicate which network to run")
    arg_parser.add_argument("learning_rate", help="Learning Rate parameter")
    arg_parser.add_argument("iterations", help="Number of iterations to perform")
    arg_parser.add_argument("batches", help="Number of batches to use")
    arg_parser.add_argument("seed", help="Seed to initialize the network")

    args = arg_parser.parse_args()

    combination = check_param_is_int("combination", args.combination)
    learning_rate = check_param_is_float("learning_rate", args.learning_rate)
    epochs = check_param_is_int("epochs", args.iterations)
    batches = check_param_is_int("batches", args.batches)
    seed = check_param_is_int("seed", args.seed)

    mlp_network(combination, learning_rate, epochs, batches, seed)