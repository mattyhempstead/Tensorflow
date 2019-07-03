import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# print("tf version", tf.__version__)

# Seed setting doesn't seem to work at all :(
# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0);
# tf.compat.v1.random.set_random_seed(0)

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


mnist = keras.datasets.cifar100.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

# Normalise pixel values
trainImages = trainImages / 255
testImages = testImages / 255

# print(mnist)

# print(trainImages[0])

# print(len(trainImages), len(testImages))
# print(trainLabels)

# print(trainLabels[0], trainImages[0])


# print(trainImages[0])



def getConvModel():
    model = keras.Sequential()

    model.add(keras.layers.Conv2D(32, (3,3), activation=tf.nn.relu, input_shape=(32,32,3), strides=(1,1)))
    model.add(keras.layers.Dropout(0.2, seed=0))

    model.add(keras.layers.Conv2D(64, (3,3), activation=tf.nn.relu, strides=(2,2)))
    model.add(keras.layers.Dropout(0.35, seed=0))

    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(128, (3,3), activation=tf.nn.relu, strides=(2,2)))
    model.add(keras.layers.Dropout(0.35, seed=0))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.35, seed=0))

    model.add(keras.layers.Dense(256, activation=tf.nn.relu))
    model.add(keras.layers.Dropout(0.35, seed=0))

    model.add(keras.layers.Dense(100, activation=tf.nn.softmax))

    return model


def getOtherModel():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    # model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(100, activation='softmax'))
    return model


model = getOtherModel()


model.compile(
    optimizer='adam', #tf.keras.optimizers.Adam(0.001)
    loss=keras.losses.sparse_categorical_crossentropy,
    # loss=keras.losses.binary_crossentropy,
    # loss=keras.losses.categorical_crossentropy,
    # loss = keras.losses.mse,
    metrics=['accuracy'],
)

model.summary()


# sys.exit()



epochs = 100


trainResults = model.fit(
    trainImages, 
    trainLabels,
    epochs = epochs,
    batch_size = 256,
    validation_data = (testImages, testLabels),
)


# print(trainResults.history)
# print(trainResults.history['accuracy'])
# print(trainResults.history['val_accuracy'])

plt.plot(
    range(epochs),
    trainResults.history['accuracy'],
    label='accuracy'
)
plt.plot(
    range(epochs),
    trainResults.history['val_accuracy'],
    label='val_accuracy'
)
plt.ylim(0,1)
# plt.xticks(range(epochs))
plt.legend()
plt.show()


testLoss, testAcc = model.evaluate(testImages, testLabels)

print("Correct: {}%".format(round(100*testAcc,2)))
print("Error: {}%".format(round(100*(1-testAcc),2)))

if input("Save model? ").upper() == "Y":
    model.save("models/model-conv.h5")
    print("Saved model as model-conv.h5")

print("Done")
