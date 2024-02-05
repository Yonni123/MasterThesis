import numpy as np
import cv2
from keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, MaxPooling2D, Conv2D, BatchNormalization, Flatten, Dense, Dropout, \
    Reshape, GaussianNoise
from tensorflow.keras.layers import *
import tensorflow as tf
from tensorflow.keras.initializers import RandomNormal

tf.random.set_seed(2)  # For reproducibility
np.random.seed(20)


def lightweight_cnn_1(input_shape=(224, 224, 3)):
    model = Sequential(name='lightweight_cnn_1')

    model.add(DepthwiseConv2D((3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))
    model.add(Dropout(0.25))

    for _ in range(3):
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # Reduced filters, efficient activation
        model.add(BatchNormalization())

    model.add(GaussianNoise(stddev=0.1))

    model.add(Dropout(0.4))
    model.add(Conv2DTranspose(3, (3, 3), strides=2, activation='relu', padding='same'))  # Example upsampling

    return model


def lightweight_cnn_2(input_shape=(224, 224, 3)):
    model = Sequential(name='lightweight_cnn_2')

    model.add(DepthwiseConv2D((3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(2, 2))

    for _ in range(3):
        model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
        model.add(BatchNormalization())

    model.add(MaxPooling2D(2, 2))

    model.add(Conv2DTranspose(64, (4, 4), strides=4, activation='relu', padding='same'))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))

    model.add(GaussianNoise(stddev=0.1))
    model.add(Dropout(0.2))

    model.add(Conv2D(3, (1, 1), activation='relu', padding='same'))

    return model


# Paper: Obfuscation arch.
def arch_1(input_shape=(224, 224, 3)):
    model = Sequential()

    # Add layers sequentially
    model.add(Conv2D(30, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))  # Adjust dropout rate as needed
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Adjust dropout rate as needed
    model.add(Dense(224 * 224 * 3, activation='relu'))
    model.add(Reshape((224, 224, 3)))  # Reshape to desired output shape
    model.add(Activation('relu'))  # Ensure pixel values in range 0-1
    return model


def currently_testing(input_shape=(224, 224, 3)):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dense(224 * 224 * 3, activation='relu'))
    model.add(Reshape(input_shape))
    return model


def up_sample(input_shape):
    model = Sequential()
    model._name = 'up-sample.V1.0'

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(9, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 112x112x27
    model.add(Conv2D(6, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 56x56x9
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 28x28x3

    # Add a dense layer
    model.add(Flatten())
    model.add(Dense(28 * 28 * 3, activation='relu'))
    model.add(Reshape((28, 28, 3)))

    # Add an up-scaling layer to increase the size of the output to the original image size
    model.add(UpSampling2D(size=(2, 2)))
    model.add(UpSampling2D(size=(2, 2)))
    model.add(UpSampling2D(size=(2, 2)))

    return model


def create_autoencoder_1(input_shape):
    model = Sequential(name='auto_encode_v1_1')

    # Encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    # Output layer
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    return model


def create_autoencoder(input_shape=(224, 224, 3)):
    model = Sequential()

    # Encoder
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2), padding='same'))

    # Decoder
    model.add(Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    # Output layer
    model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))

    return model


m = up_sample((224, 224, 3))
m.summary()
