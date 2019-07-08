import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
tf.set_random_seed(0)


print("tf version", tf.__version__)


mnist = keras.datasets.mnist.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

# print(mnist)

# print(trainImages[0])

print(len(trainImages), len(testImages))
print(trainLabels)

# plt.figure(figsize=(8,8))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(trainImages[i], cmap='gray')
#     plt.xlabel(trainLabels[i])
# plt.show()



# model = keras.Sequential([
#     keras.layers.Flatten(input_shape=(28, 28)),
#     keras.layers.Dense(256, activation=tf.nn.relu),
#     keras.layers.Dense(128, activation=tf.nn.relu),
#     keras.layers.Dense(64, activation=tf.nn.relu),
#     keras.layers.Dense(10, activation=tf.nn.softmax)
# ])

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(800, activation=tf.nn.relu))
model.add(keras.layers.Dense(800, activation=tf.nn.relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer='adam', 
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(trainImages, trainLabels, epochs=3)



test_loss, test_acc = model.evaluate(testImages, testLabels)

# print('Test accuracy:', test_acc)