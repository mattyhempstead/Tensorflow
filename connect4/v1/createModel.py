import tensorflow as tf
from tensorflow import keras

def createModel():
    model = keras.Sequential()

    # model.add(keras.layers.Conv2D(128, (2, 2), activation='tanh', input_shape=(7, 6, 1)))
    # model.add(keras.layers.Conv2D(128, (2, 2), activation='tanh'))
    # model.add(keras.layers.Conv2D(256, (2, 2), activation='tanh'))
    # model.add(keras.layers.Conv2D(256, (2, 2), activation='tanh'))

    model.add(keras.layers.Conv2D(128, (3, 3), activation='tanh', input_shape=(7, 6, 1)))
    model.add(keras.layers.Conv2D(256, (3, 3), activation='tanh'))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(256, activation='tanh'))
    # model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    # model.add(keras.layers.Dropout(0.1))

    model.compile(
        optimizer='adam', #tf.keras.optimizers.Adam(0.001)
        loss=keras.losses.binary_crossentropy,
        metrics=[],
    )
    return model