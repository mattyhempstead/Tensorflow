import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

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


model = keras.models.load_model("models/model-conv.h5")
model.summary()

# Get model accuracy
model.evaluate(testImages, testLabels)



predictions = model.predict(testImages)
predictedLabels = [np.argmax(i) for i in predictions]

correctCount = sum([i==k for i,k in zip(predictedLabels, testLabels)])
print(correctCount, len(testLabels))



# Get a list of all the incorrect images
incorrect = [
    (image, trueLabel, predictedLabel, certainty[predictedLabel]) 
    for image, trueLabel, predictedLabel, certainty
    in zip(testImages, testLabels, predictedLabels, predictions)
    if trueLabel != predictedLabel
]

print([(i[1],i[2],i[3]) for i in incorrect])


plt.figure(figsize=(16,8))
for i in range(min(60,len(incorrect))):
    plt.subplot(6,10,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(np.reshape(incorrect[i][0], (28,28)), cmap='gray')
    plt.xlabel(
        "{} - {}% ({})".format(
            incorrect[i][1],
            round(100*incorrect[i][3], 2),
            incorrect[i][2]
        )
    )
plt.subplots_adjust(hspace=0.5, wspace=0.5)     # Add vertical spacing between plots
plt.show()
plt.subplot()

