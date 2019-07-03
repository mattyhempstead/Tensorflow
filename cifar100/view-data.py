import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt


mnist = keras.datasets.cifar100.load_data(label_mode="fine")
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255


def getImageNames():
    import pickle
    with open('meta', 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return [i.decode('UTF-8') for i in dict[b'fine_label_names']]

imageNames = getImageNames()



plt.figure(figsize=(16,8))
for i in range(50):
    plt.subplot(5,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(trainImages[i], cmap='gray')
    plt.xlabel(imageNames[trainLabels[i][0]])
plt.subplots_adjust(hspace=0.5)     # Add vertical spacing between plots
plt.show()
plt.subplot()
