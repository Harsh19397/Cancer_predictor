import tensorflow as tf
import keras
from tensorflow.keras import Sequential
import numpy as np


#Artificial Neural Network
def ANN(X_train, y_train, X_test, y_test):
    model = Sequential()
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(40, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(10, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    X_train_fit = np.expand_dims(X_train, axis=2)
    X_test_fit = np.expand_dims(X_test, axis=2)
    model.fit(X_train, y_train, epochs = 100, validation_data=(X_test, y_test))
    accuracy_NN = model.evaluate(X_test, y_test)
    model.save('Cancer_predictor_nn.h5')
    
    return accuracy_NN
    
    