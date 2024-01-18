import cv2
from keras.utils import load_img
from matplotlib import pyplot as plt
from tensorflow.keras.models import Sequential, load_model, Model, clone_model

import Utils as utils

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)   # For reproducibility
np.random.seed(1)


num_classes = 1000  # Number of classes in the dataset
input_shape = (224, 224, 3)

inference_model = utils.get_ResNet50()  # Load the pre-trained ResNet50 model
obfmodel = utils.get_obfmodel(input_shape, num_neuron=100)
combined_model = Model(inputs=obfmodel.input, outputs=inference_model(obfmodel.output))
combined_model.load_weights('models/mnist/combined-model.h5')

img_path = '../Random_Images/MNIST/3.jpg'
print("Combined prediction: ", predict_image(img_path, combined_model))

obfnet = extract_obfnet(combined_model)
obf_image = obfuscate_image(img_path, obfnet)
plt.imshow(obf_image, cmap='gray')
plt.title('Obfuscated Image')
plt.show()
cv2.imwrite('../Random_Images/MNIST/obfuscated3.jpg', obf_image)

infnet = extract_infnet(combined_model)
print("Obfuscate prediction: ", predict_image('../Random_Images/MNIST/obfuscated3.jpg', infnet))