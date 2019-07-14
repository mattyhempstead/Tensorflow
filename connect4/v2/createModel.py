import tensorflow as tf
from tensorflow import keras


def createModel(modelType="conv"):
 
    if modelType == "conv":
        model = createModelConv()
    elif modelType == "dense":
        model = createModelDense()

    model.compile(
        # optimizer='adam',
        optimizer = tf.keras.optimizers.Adam(),
        loss=keras.losses.binary_crossentropy,
        metrics=[],
    )
    return model



def createModelConv():
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

    return model



def createModelDense():
    model = keras.Sequential()
    
    model.add(keras.layers.Flatten(input_shape=(7, 6, 1)))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(512, activation='tanh'))
    model.add(keras.layers.Dense(256, activation='tanh'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # model.add(keras.layers.Dropout(0.1))

    return model
