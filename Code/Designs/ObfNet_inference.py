import numpy as np
import cv2
from keras.layers import *
from keras.utils import load_img
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
from tensorflow.keras.layers import *
from tensorflow.keras import backend as K
from keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.initializers import RandomNormal
import tensorflow as tf
from matplotlib import pyplot as plt
import cv2
from Code.Helper import Utils
from Code import PATHS
import matplotlib.pyplot as plt
from matplotlib.image import imread
from Code.Designs import cnn, Diffused_ObfNet


def plot_image_array_gray(images, labels):
    for i in range(0, len(labels)):
        ax = plt.subplot(1, len(labels), i + 1)
        plt.axis('off')
        plt.text(0.5, -0.1, labels[i], horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
        plt.imshow(images[i], cmap=plt.cm.gray)

    plt.tight_layout()
    plt.show()


model_paths = {
    'model_1': r'C:\MT\Code\History\session_20240204_001027\\best_obf_weights.h5',
    'model_2': r'C:\MT\Code\History\session_20240203_140656_84%\\best_obf_weights.h5',
    'model_3': r'C:\MT\Code\History\session_20240204_191415\\best_obf_weights.h5',
    'model_4': r'C:\MT\Code\History\session_20240203_004732_92%\best_obf_weights.h5'
}


# Function to load a specific model
def load_selected_model(model_dict, selected_key):
    model_path = model_dict.get(selected_key)
    if model_path:
        return load_model(model_path)
    else:
        raise ValueError(f'Model with key "{selected_key}" not found in the model dictionary.')


# Example usage
selected_model_key = 'model_1'
model = load_selected_model(model_paths, selected_model_key)

#model = Diffused_ObfNet.diffused_ObfNet((224, 224, 3))
orgImg = np.array(load_img('C:\MT\Code\Images\ImageNet\\img3.JPEG'))

img = preprocess_input(orgImg)
img = np.expand_dims(img, axis=0)

output = model.predict(img)
resnet = Utils.get_pretrained_ResNet50(PATHS.RESNET_WEIGHTS)
otClass = resnet.predict(output)
preClass = np.argmax(otClass[0])
print(preClass)
output = np.squeeze(output, axis=0)
output = 255 * (output / np.max(output))
# output *= 255
img = np.squeeze(img, axis=0)

plot_image_array_gray((orgImg, img, output), ("Original", "Preprocessed", "Obfuscated"))
