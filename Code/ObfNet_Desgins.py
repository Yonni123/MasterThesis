import numpy as np
import cv2
from keras.layers import *
from keras.utils import load_img
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras.applications.resnet import ResNet50
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
tf.random.set_seed(2)   # For reproducibility
np.random.seed(20)


def get_resize_layer(target_shape):
    layer = Lambda(lambda x: tf.image.resize(x, (target_shape[0], target_shape[1])))
    layer._name = 'up-sample_layer'
    return layer

def conv_reshape(input_shape):
    model = Sequential()
    model._name = 'conv-reshape_model'

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same' ,input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(192, kernel_size=(3, 3), activation='softmax', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output and add a dense layer
    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Reshape(input_shape))
    return model

def up_scaled(input_shape):
    model = Sequential()
    model._name = 'up-scaled_model'

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a dense layer
    model.add(Flatten())
    model.add(Dense(26*26*3, activation='relu', kernel_initializer=RandomNormal(mean=0.0, stddev=0.01, seed=69)))
    model.add(Reshape((26, 26, 3)))

    # Add an up-scaling layer to increase the size of the output to the original image size
    resize_layer = get_resize_layer(target_shape=input_shape)
    #model.add(resize_layer)

    return model

def deconv(input_shape):
    model = Sequential()

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(27, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 112x112x27
    model.add(Conv2D(9, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 56x56x9
    model.add(Conv2D(3, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))   # 28x28x3

    # Add a dense layer
    model.add(Flatten())
    model.add(Dense(28*28*3, activation='relu'))

    # Add deconvolutional layers to increase the size of the output to the original image size
    model.add(Reshape((28, 28, 3)))  # Reshape back to the spatial dimensions

    model.add(Conv2DTranspose(3, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))  # 56x56x3
    model.add(Conv2DTranspose(9, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))  # 112x112x9
    model.add(Conv2DTranspose(27, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D(size=(2, 2)))  # 224x224x27

    # Output layer
    model.add(Conv2DTranspose(3, kernel_size=(1, 1), activation='sigmoid', padding='same'))

    return model


m = deconv((224, 224, 3))
m.summary()

# quit()

img = cv2.imread('Random_Images/lion.jpg')
img = cv2.resize(img, (224, 224))
cv2.imshow('Original Image', img)
img_array = np.array(img)
img_array_channelfirst = np.moveaxis(img_array, -1, 0)

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array.astype('float32')
img_array /= 255

# Add random weights to the model
for layer in m.layers:
    print(layer.name)
    print(layer.get_weights())
    random_weights = []
    for weight in layer.get_weights():
        random_weights.append(np.random.rand(*weight.shape))
    layer.set_weights(random_weights)
    print('After:')
    print(layer.get_weights())

# Make predictions
res = m.predict(img_array)
res = np.squeeze(res, axis=0)
res_channelfirst = np.moveaxis(res, 0, -1)
res *= 255

res = res.astype('uint8')
cv2.imshow('Resized Image', res)
cv2.waitKey(0)
