import os
import sys

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

current_directory = os.getcwd()
parent_directory = os.path.dirname(current_directory)
items = os.listdir(parent_directory)
folders = [item for item in items if os.path.isdir(os.path.join(parent_directory, item))]
for folder in folders:
    full_path = os.path.join(parent_directory, folder)
    sys.path.append(full_path)
parent_directory = os.path.dirname(current_directory)
grandparent_directory = os.path.dirname(parent_directory)
sys.path.append(parent_directory)
sys.path.append(grandparent_directory)
import Train
from Helper import Utils
import PATHS
from Designs.cnn import lightweight_cnn_1, lightweight_cnn_2, \
    arch_1, currently_testing, up_sample, create_autoencoder_1
from Designs.ObfNet_Desgins import *
from Designs.Diffused_ObfNet import diffused_ObfNet
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model, Model, clone_model
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
import pickle as pkl


com_path = 'models/asl/combined-model.h5'
combined_model = load_model(com_path)
obf = Sequential()
for i in range(len(combined_model.layers)):
    if combined_model.layers[i] == combined_model.layers[-1]:
        continue
    obf.add(combined_model.layers[i])
obf.summary()

with open(r'C:\train.pkl', "rb") as f:
    X_train, y_train = pkl.load(f)
X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.1)
test_img = X_test[-1]
cv2.imwrite('aa.jpg', test_img)
img = np.expand_dims(test_img, axis=0)
output = obf.predict(img)
output = np.squeeze(output, axis=0)
# Split into individual channels
channel_0 = output[:, :, 0]
channel_1 = output[:, :, 1]
channel_2 = output[:, :, 2]

normalized_channel_0 = (channel_0 / np.max(channel_0))*255
normalized_channel_1 = (channel_1 / np.max(channel_1))*255
normalized_channel_2 = (channel_2 / np.max(channel_2))*255

normalized_img = np.stack([normalized_channel_0, normalized_channel_1, normalized_channel_2], axis=-1).astype(np.uint8)

cv2.imwrite('bb.jpg', normalized_img)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the first image on the first subplot
axes[0].imshow(test_img)
axes[0].set_title('Original image')

# Display the second image on the second subplot
axes[1].imshow(normalized_img)
axes[1].set_title('Obfuscated image')

# Adjust layout and show the figure
plt.tight_layout()
plt.show()