import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.compat.v1.random.set_random_seed(0)


print("tf version", tf.__version__)


mnist = keras.datasets.mnist.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

# print(mnist)

# print(trainImages[0])

# print(len(trainImages), len(testImages))
# print(trainLabels)



model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dropout(0.1))

model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(256, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

model.fit(trainImages, trainLabels, epochs=20)

test_loss, test_acc = model.evaluate(testImages, testLabels)
print("Model accuracy {}".format(test_acc))


model.save("models/model.h5")
