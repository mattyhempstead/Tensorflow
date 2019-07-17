import tensorflow as tf
from tensorflow import keras


def createModel():
 
    model = keras.Sequential()

    model.add(keras.layers.Flatten(input_shape=(49,)))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(2, activation='sigmoid'))

    model.compile(
        optimizer = tf.keras.optimizers.Adam(),
        loss = keras.losses.binary_crossentropy,
        metrics = [],
    )
    return model

