import os, sys, random
import tensorflow as tf
from tensorflow import keras
import numpy as np
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



mnist = keras.datasets.cifar10.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

# "car" should actually be "automobile" and "plane" should be "airplane" (replaced for text space)
imageNames = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


# print(mnist)

# print(trainImages[0])

# print(len(trainImages), len(testImages))
# print(trainLabels)
# print(trainImages[0])

modelFile = "model.h5" if (len(sys.argv)==1) else sys.argv[1]
model = keras.models.load_model("models/{}".format(modelFile))

model.summary()

# Get model accuracy after training
model.evaluate(testImages, testLabels)

# print('Test accuracy:', test_acc)



predictions = model.predict(testImages)



# Count the percentage of correct guesses for each class
predictedLabels = [np.argmax(i) for i in predictions]
trueLabels = [i[0] for i in testLabels]
correctLabels = [predictedLabels[i] for i in range(len(trueLabels)) if predictedLabels[i] == trueLabels[i]]
labelCount = [trueLabels.count(i) for i in range(10)]     # Number of each type of label
labelCorrectCount = [correctLabels.count(i) for i in range(10)]
labelCorrectProportion = [labelCorrectCount[i] / labelCount[i] for i in range(10)]
for i in range(10):
    print("{}: {}%".format(imageNames[i], round(100*labelCorrectProportion[i], 2)))




def plotImage(i, img, trueLabel, predictionsArray):
    predictedLabel = np.argmax(predictionsArray)
    plt.subplot(5, 10, 2*i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.imshow(img, cmap='gray')

    plt.xlabel(
        "{} {}% ({})".format(
            imageNames[predictedLabel],
            round(100*max(predictionsArray), 2),
            imageNames[trueLabel]
        ),
        color = "green" if (predictedLabel == trueLabel) else "red",
        size = 8
    )


    plt.subplot(5, 10, 2*i+2)
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    barPlot = plt.bar(range(10), predictionsArray, color="#777777")
    plt.ylim([0, 1]) 
    barPlot[predictedLabel].set_color('red')
    barPlot[trueLabel].set_color('green')



startIndex = random.randint(0,5000)
plt.figure(figsize=(16,8))
for i in range(25):
    plotImage(
        i,
        testImages[startIndex+i],
        testLabels[startIndex+i][0],
        predictions[startIndex+i]
    )

plt.subplots_adjust(hspace=0.5)     # Add vertical spacing between plots
plt.show()





