import os
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


mnist = keras.datasets.cifar10.load_data()
(trainImages, trainLabels), (testImages, testLabels) = mnist

trainImages = trainImages / 255
testImages = testImages / 255

# print(mnist)

# print(trainImages[0])

# print(len(trainImages), len(testImages))
# print(trainLabels)

# print(trainLabels[0], trainImages[0])




model = keras.Sequential()

model.add(keras.layers.Flatten(input_shape=(32,32,3)))

model.add(keras.layers.Dense(1024, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2, seed=0))

model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2, seed=0))

model.add(keras.layers.Dense(512, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2, seed=0))

model.add(keras.layers.Dense(256, activation=tf.nn.relu))
model.add(keras.layers.Dropout(0.2, seed=0))

model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(
    optimizer='adam', #tf.keras.optimizers.Adam(0.001)
    loss=keras.losses.sparse_categorical_crossentropy,
    # loss=keras.losses.binary_crossentropy,
    # loss=keras.losses.categorical_crossentropy,
    # loss = keras.losses.mse,
    metrics=['accuracy'],
)

model.summary()





# vectorTrainLabels = []
# for i in trainLabels:
#     newLabel = np.array([1 if k==i[0] else 0 for k in range(10)])
#     vectorTrainLabels.append(newLabel)
# vectorTrainLabels = np.array(vectorTrainLabels, dtype=np.uint8)

# trainLabels = vectorTrainLabels


# vectorTestLabels = []
# for i in testLabels:
#     newLabel = np.array([1 if k==i[0] else 0 for k in range(10)])
#     vectorTestLabels.append(newLabel)
# vectorTestLabels = np.array(vectorTestLabels, dtype=np.uint8)

# testLabels = vectorTestLabels


epochs = 50


trainResults = model.fit(
    trainImages, 
    trainLabels,
    epochs = epochs,
    batch_size = 512,
    validation_data = (testImages, testLabels),
    # callbacks = [tf.keras.callbacks.ModelCheckpoint(
    #     "models/cp.ckpt",
    #     save_weights_only=True,
    #     verbose=1
    # )]
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


test_loss, test_acc = model.evaluate(testImages, testLabels)

if input("Save model? ").upper() == "Y":
    model.save("models/model.h5")
    print("Saved model")

print("Done")
