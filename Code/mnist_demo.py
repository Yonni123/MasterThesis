import PATHS
import cv2
from keras.utils import load_img
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D, BatchNormalization
import Utils

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)   # For reproducibility
np.random.seed(1)

def get_inference():
    input_shape = (28, 28, 1)
    number_classes = 10
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_classes, activation='softmax'))
    return model

def get_obfmodel():
    input_shape = (28, 28, 1)
    num_neuron = 100
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(BatchNormalization())
    model.add(Reshape(input_shape))
    return model

def preprocess_MNIST(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)
    img = img.astype('float32')
    img /= 255
    return img


if __name__ == "__main__":
    # Load inference and obfnet models, combine and train them (load weights)
    inference_model = get_inference()
    obfmodel = get_obfmodel()
    combined_model = Utils.join_models(obfmodel, inference_model)
    combined_model.load_weights('History/MNIST_Demo/combined-model.h5')

    # Load a test image
    img_path = 'Images/MNIST/3.jpg'
    img = load_img(img_path, target_size=(28, 28))
    img = preprocess_MNIST(img)

    # Predict on combined model
    predictions = combined_model.predict(img, verbose=0)
    number = np.argmax(predictions[0])
    print("Combined prediction: ", number)

    # Split the combined model, obfuscate the test image and predict it using the inference model
    obfnet, infnet = Utils.split_models(combined_model)
    obf_image = obfnet.predict(img, verbose=0)
    predictions = infnet.predict(obf_image, verbose=0)
    number = np.argmax(predictions[0])
    print("Obfuscate prediction: ", number)
    predictions = infnet.predict(img, verbose=0)
    number = np.argmax(predictions[0])
    print("Pure image prediction: ", number)

    # Display the obfuscated image (normalize all values to be in 0-255 range)
    img = np.squeeze(img, 3)
    img = np.squeeze(img, 0)
    obf_image = np.squeeze(obf_image, 3)
    obf_image = np.squeeze(obf_image, 0)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    ax1.imshow(obf_image, cmap='gray')
    ax1.set_title('Obfuscated Image')
    ax2.imshow(img, cmap='gray')
    ax2.set_title('Original Image')
    plt.tight_layout()
    plt.show()