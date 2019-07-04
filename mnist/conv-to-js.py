import os, sys
import random
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflowjs as tfjs

model = keras.models.load_model("models/model-conv.h5")
model.summary()

tfjs.converters.save_keras_model(model, "models")

