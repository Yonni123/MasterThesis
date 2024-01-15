import cv2
from keras.utils import load_img
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)   # For reproducibility
np.random.seed(1)

def get_inference(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model

def get_obfmodel(input_shape, num_neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(Reshape(input_shape))
    return model

def predict_image(path, model):
    size = (model.input_shape[1], model.input_shape[2])
    img = load_img(path, target_size=size)
    x = np.array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=3)
    x = x.astype('float32')
    x /= 255
    predictions = model.predict(x)
    return np.argmax(predictions[0])

def extract_obfnet(model):
    split_layer_index = len(model.layers) - 1   # The last layer is the Sequential layer which is the inference model

    # Split the combined model into inference and obfuscation models
    layers = model.layers[:split_layer_index]

    # Extract input and output tensors
    input_tensor = layers[0].input
    output_tensor = layers[-1].output

    # Create the obfuscation model
    obfnet = Model(inputs=input_tensor, outputs=output_tensor)

    # Transfer weights of the layers
    for i in range(len(obfnet.layers)):
        obfnet.layers[i].set_weights(model.layers[i].get_weights())

    return obfnet

def extract_infnet(model):
    split_layer_index = len(model.layers) - 1   # The last layer is the Sequential layer which is the inference model

    # Check if the last layer is Sequential
    last_layer = model.layers[-1]
    if isinstance(last_layer, Sequential):
        # If so, extract the inference model from the Sequential layer
        infnet = Sequential()
        for layer in last_layer.layers:
            infnet.add(layer)

        # Transfer weights of the layers
        for i in range(len(infnet.layers)):
            infnet.layers[i].set_weights(last_layer.layers[i].get_weights())

        return infnet
    else:
        raise Exception('The last layer is not Sequential')


def obfuscate_image(path, model):
    size = (model.input_shape[1], model.input_shape[2])
    img = load_img(path, target_size=size)
    x = np.array(img)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    x = np.expand_dims(x, axis=0)
    x = np.expand_dims(x, axis=3)
    x = x.astype('float32')
    x /= 255
    obf_image = model.predict(x)[0]
    obf_image = np.squeeze(obf_image, axis=2)
    obf_image *= 255
    obf_image = obf_image.astype('uint8')
    return obf_image


num_classes = 10
input_shape = (28, 28, 1)

inference_model = get_inference(input_shape, num_classes)
obfmodel = get_obfmodel(input_shape, 100)
combined_model = Model(inputs=obfmodel.input, outputs=inference_model(obfmodel.output))
combined_model.load_weights('models/mnist/combined-model.h5')

img_path = '../Random_Images/MNIST/3.jpg'
print("Prediction: ", predict_image(img_path, combined_model))

obfnet = extract_obfnet(combined_model)
obf_image = obfuscate_image(img_path, obfnet)
plt.imshow(obf_image, cmap='gray')
plt.title('Obfuscated Image')
plt.show()

infnet = extract_infnet(combined_model)
print("Prediction: ", predict_image(img_path, infnet))