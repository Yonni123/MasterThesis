import numpy as np
import cv2
from keras.layers import Lambda
from keras.utils import load_img
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from keras.applications.resnet import ResNet50
import tensorflow as tf
tf.random.set_seed(69)   # For reproducibility


def get_resize_layer(target_shape):
    layer = Lambda(lambda x: tf.image.resize(x, (target_shape[0], target_shape[1])))
    layer._name = 'up-sample_layer'
    return layer

def conv_reshape(input_shape):
    model = Sequential()

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same' ,input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(96, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(192, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flatten the output and add a dense layer
    model.add(Flatten())
    model.add(Dropout(0.25))

    model.add(Reshape(input_shape))
    return model

def up_scaled(input_shape):
    model = Sequential()

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a dense layer
    model.add(Flatten())
    model.add(Dense(26*26*3, activation='relu'))
    model.add(Reshape((26, 26, 3)))

    # Add an up-scaling layer to increase the size of the output to the original image size
    resize_layer = get_resize_layer(target_shape=input_shape)
    model.add(resize_layer)

    return model

def deconv(input_shape):
    model = Sequential()

    # 3 convolutional layers to reduce the image size from 224x224 to 112x112 to 56x56 to 28x28
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Add a dense layer
    model.add(Flatten())
    model.add(Dense(26*26*3, activation='relu'))
    model.add(Reshape((26, 26, 3)))

    # Add an up-scaling layer to increase the size of the output to the original image size
    resize_layer = get_resize_layer(target_shape=input_shape)
    model.add(resize_layer)

    return model


m = up_scaled((224, 224, 3))
m.summary()

quit()

img = cv2.imread('Random_Images/lion.jpg')
img = cv2.resize(img, (224, 224))
cv2.imshow('Original Image', img)

img_array = np.array(img)

# Give the model random weights and biases
for i in range(len(m.layers)):
    m.layers[i].set_weights([np.random.rand(*w.shape) for w in m.layers[i].get_weights()])

img_array = np.expand_dims(img_array, axis=0)
img_array = img_array.astype('float32')
res = m.predict(img_array)
res = np.squeeze(res, axis=0)
res = res.astype('uint8')
cv2.imshow('Resized Image', res)
cv2.waitKey(0)
