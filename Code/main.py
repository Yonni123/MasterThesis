import cv2
from keras.utils import load_img
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, Activation

import Utils as utils

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)   # For reproducibility
np.random.seed(1)


def aaaaa(target_dims, num_neuron):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=(target_dims)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_neuron))

    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model

m = aaaaa((28, 28, 3), 1000)
m.summary()