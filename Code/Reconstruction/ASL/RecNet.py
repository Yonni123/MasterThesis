import os
import sys

import numpy as np
from keras.callbacks import ModelCheckpoint
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


def get_obfmodel_mlp(target_dims, num_neuron=512):
    model = Sequential()
    model.add(Flatten(input_shape=(target_dims)))
    model.add(Dense(num_neuron))
    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model


def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return np.array(images)


com_path = 'models/asl/combined-model.h5'
combined_model = load_model(com_path)
obf_model = Sequential()
for i in range(len(combined_model.layers)):
    if combined_model.layers[i] == combined_model.layers[-1]:
        continue
    obf_model.add(combined_model.layers[i])
obf_model.summary()

train_rec = False
if train_rec:
    train_path = r'C:\Users\yoyo\Desktop\EXJOBB\MasterThesis\Code\Train\0'
    train_raw = load_images_from_folder(train_path)
    test_path = r'C:\Users\yoyo\Desktop\EXJOBB\MasterThesis\Code\test\0'
    test_raw = load_images_from_folder(test_path)
    print("Shape of loaded train images array:", train_raw.shape)
    print("Shape of loaded test images array:", test_raw.shape)
    train_obf = obf_model.predict(train_raw)
    test_obf = obf_model.predict(test_raw)
    print("Shape of train_obf images array:", train_obf.shape)
    print("Shape of test_obf images array:", test_obf.shape)

    RecNet = get_obfmodel_mlp((64, 64, 3), num_neuron=1024)
    RecNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    RecNet.fit(
        x=train_obf,  # Input data (obfuscated images)
        y=train_raw,  # Target data (original images)
        epochs=1000,
        validation_data=(test_obf, test_raw),
        callbacks=[ModelCheckpoint('RecNet.h5',
                                   monitor='val_accuracy',
                                   verbose=1,
                                   save_best_only=True,
                                   mode='max')]
    )
    quit()

RecNet = load_model('RecNet.h5')
orig_test_img = np.array(load_img(r'C:\Users\yoyo\Desktop\EXJOBB\MasterThesis\Code\Images\ASL\33.png', target_size=(64, 64)))
test_img = np.expand_dims(orig_test_img, axis=0)
rec_img = RecNet.predict(test_img)
rec_img = np.squeeze(rec_img, axis=0)
rec_img = ((rec_img - np.min(rec_img)) / (np.max(rec_img) - np.min(rec_img)) * 255).astype(np.uint8)
rec_img = cv2.cvtColor(rec_img, cv2.COLOR_BGR2RGB)
cv2.imwrite(r'C:\Users\yoyo\Desktop\EXJOBB\MasterThesis\Code\Images\ASL\rec.png', rec_img)
cv2.imwrite(r'C:\Users\yoyo\Desktop\EXJOBB\MasterThesis\Code\Images\ASL\input.png', orig_test_img)
