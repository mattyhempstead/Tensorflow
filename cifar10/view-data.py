import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.cifar10.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

imageNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "boat", "truck"]



plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap='gray')
    plt.xlabel(imageNames[trainLabels[i][0]])
plt.subplots_adjust(hspace=0.5)     # Add vertical spacing between plots
plt.show()
plt.subplot()
