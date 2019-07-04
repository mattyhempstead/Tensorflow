import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# print("tf version", tf.__version__)


EPOCHS = 30


# Seed setting doesn't seem to work at all :(
# os.environ['PYTHONHASHSEED'] = '0'
# random.seed(0)
# np.random.seed(0)
# tf.random.set_seed(0);
# tf.compat.v1.random.set_random_seed(0)

# Stop some of the random logging
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Get mnist data
mnist = keras.datasets.mnist.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

# Normalise pixel values
trainImages = trainImages / 255
testImages = testImages / 255

# Reshape for conv net
testImages = np.reshape(testImages, (10000, 28, 28, 1))
trainImages = np.reshape(trainImages, (60000, 28, 28, 1))


def getConvModel():
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    # model.add(keras.layers.Dropout(0.05))
    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))
    model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    # model.add(keras.layers.MaxPooling2D((2, 2)))
    # model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(512, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(10, activation='softmax'))
    return model



model = getConvModel()


model.compile(
    optimizer='adam', #tf.keras.optimizers.Adam(0.001)
    loss=keras.losses.sparse_categorical_crossentropy,
    metrics=['accuracy'],
)

model.summary()

# sys.exit()



trainResults = model.fit(
    trainImages, 
    trainLabels,
    epochs = EPOCHS,
    batch_size = 256,
    validation_data = (testImages, testLabels),
)


# print(trainResults.history)
# print(trainResults.history['accuracy'])
# print(trainResults.history['val_accuracy'])

plt.plot(
    range(EPOCHS),
    trainResults.history['accuracy'],
    label='accuracy'
)
plt.plot(
    range(EPOCHS),
    trainResults.history['val_accuracy'],
    label='val_accuracy'
)
plt.ylim(0,1)
# plt.xticks(range(EPOCHS))
plt.legend()
plt.show()


testLoss, testAcc = model.evaluate(testImages, testLabels)

print("Correct: {}%".format(round(100*testAcc,2)))
print("Error: {}%".format(round(100*(1-testAcc),2)))

if input("Save model? ").upper() == "Y":
    model.save("models/model-conv.h5")
    print("Saved model as model-conv.h5")

print("Done")
