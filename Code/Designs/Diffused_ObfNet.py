import numpy as np
import cv2
from keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Dense, Dropout, \
    Reshape
from tensorflow.keras.layers import *
import tensorflow as tf

tf.random.set_seed(2)  # For reproducibility
np.random.seed(20)


def diffused_ObfNet(input_shape=(224, 224, 3)):
    model = Sequential(name='gtp_V.0')

    # Feature Extractor
    model.add(Conv2D(16, (3, 3), padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D((2, 2)))

    # Obfuscation Block 1
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Obfuscation Block 2
    model.add(Conv2D(32, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    # Decoder Block
    model.add(Conv2DTranspose(16, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(UpSampling2D((2, 2)))

    # Output Layer
    model.add(Conv2D(3, (224, 224), activation='relu', padding='same'))

    return model


def lightweight_cnn_1(input_shape=(224, 224, 3)):
    model = Sequential(name='lightweight_cnn_1')

    model.add(DepthwiseConv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    for _ in range(2):
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Reduced filters, efficient activation
        model.add(BatchNormalization())

    model.add(Conv2DTranspose(3, (3, 3), strides=2, activation='relu', padding='same'))  # Example upsampling

    return model


m = diffused_ObfNet((224, 224, 3))
m.summary()
