import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import os
# print("tf version", tf.__version__)


np.random.seed(0)
tf.compat.v1.random.set_random_seed(0)


mnist = keras.datasets.mnist.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

# print(mnist)

# print(trainImages[0])

# print(len(trainImages), len(testImages))
# print(trainLabels)
# print(trainImages[0])




model = keras.models.load_model("models/model.h5")

model.summary()

# Get model accuracy after training
model.evaluate(testImages, testLabels)

# print('Test accuracy:', test_acc)



predictions = model.predict(testImages)

plt.figure(figsize=(8,8))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(testImages[i], cmap='gray')
    plt.xlabel(
        "{} - {}% ({})".format(
            np.argmax(predictions[i]),
            round(100*max(predictions[i]), 2),
            testLabels[i]
        )
    )
plt.subplots_adjust(hspace=0.5)     # Add vertical spacing between plots
plt.show()
plt.subplot()

