# This is the original source code for ObfNet taken from:
# https://github.com/ntu-aiot/ObfNet/blob/master/mnist/main.py

from __future__ import print_function

from keras.utils import load_img
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Reshape, Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import cv2
import click

import tensorflow as tf
import numpy as np
tf.random.set_seed(1)
np.random.seed(1)


def get_inference_cnn(input_shape, num_classes):
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
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_inference_mlp(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_obfmodel_mlp(input_shape, num_neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(Reshape(input_shape))
    return model


def get_obfmodel_cnn(input_shape, num_neuron=32):
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
    model.add(Reshape(input_shape))
    return model


def get_mid_out(model, layer_num, data):
    get_output = K.function([model.layers[0].input],
                            [model.layers[layer_num].output])
    return get_output([data])[0]


def main(is_inf_cnn, is_obf_cnn, is_rec_cnn, obf_num_neuron, rec_num_neuron, train_obf=True, train_inf=True, train_rec=True, save_folder='a'):

    os.makedirs("models/mnist/", exist_ok=True)

    batch_size = 128
    num_classes = 10
    epochs = 14

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    if is_inf_cnn:
        inference_model = get_inference_cnn(input_shape, num_classes)
    else:
        inference_model = get_inference_mlp(input_shape, num_classes)
    inference_model.compile(loss='categorical_crossentropy',
                            optimizer='adam',
                            metrics=['accuracy'])
    if train_inf:
        print('Training InfNet')
        inference_model.fit(x_train,
                            y_train,
                            batch_size=batch_size,
                            epochs=epochs,
                            callbacks=[
                                ModelCheckpoint('models\mnist\inf.h5',
                                                monitor='val_accuracy',
                                                verbose=1,
                                                save_best_only=True,
                                                mode='max')
                            ],
                            verbose=1,
                            validation_data=(x_test, y_test))
    else:
        print('Trying to load InfNet weights')
        inference_model.load_weights('models\mnist\inf.h5')

    print('InfNet test result: ')
    score = inference_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print()

    ######################
    # training of ObfNet #
    ######################

    if is_obf_cnn:
        obfmodel = get_obfmodel_cnn(input_shape, obf_num_neuron)
    else:
        obfmodel = get_obfmodel_mlp(input_shape, obf_num_neuron)
        obfmodel.build((None, 28, 28, 1))

    inference_model.trainable = False
    for l in inference_model.layers:
        l.trainable = False

    combined_model = Model(inputs=obfmodel.input,
                           outputs=inference_model(obfmodel.output))
    combined_model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    if train_obf:
        print('Training ObfNet')
        combined_model.fit(x=x_train,
                           y=y_train,
                           batch_size=batch_size,
                           epochs=epochs,
                           callbacks=[
                               ModelCheckpoint('models/mnist/combined-model.h5',
                                               monitor='val_accuracy',
                                               verbose=1,
                                               save_best_only=True,
                                               mode='max')
                           ],
                           verbose=1,
                           validation_data=(x_test, y_test))
    else:
        print('Trying to load InfNet weights')
        combined_model.load_weights('models\mnist\combined-model.h5')

    ######################
    # testing of ObfNet #
    ######################

    print('ObfNet test result: ')
    score = combined_model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    print()

    # Save example images from the paper (to compare with reconstructed images later)
    os.makedirs("imgs/mnist/" + save_folder + '/', exist_ok=True)
    ind = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
    h1 = []
    for i in ind:
        h1.append(x_test[i] * 255)
    v1 = np.concatenate(np.array(h1), axis=1)
    cv2.imwrite("imgs/mnist/" + save_folder + '/input.jpg', v1)
    midout = get_mid_out(combined_model, -2, x_test)
    h = []
    for i in ind:
        h.append(midout[i] * 255)
    v = np.concatenate(np.array(h), axis=1)
    cv2.imwrite("imgs/mnist/" + save_folder + '/obfresult.jpg', v)

    ##########################
    # Reconstruction attack  #
    ##########################
    print('Reconstruction attack!')
    # Create training and testing data by obfuscating x_train and x_test
    x_train_obf = get_mid_out(combined_model, -2, x_train)
    x_test_obf = get_mid_out(combined_model, -2, x_test)

    if is_rec_cnn:
        RecNet = get_obfmodel_cnn(input_shape, rec_num_neuron)
    else:
        RecNet = get_obfmodel_mlp(input_shape, rec_num_neuron)
        RecNet.build((None, 28, 28, 1))
    RecNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    if train_rec:
        print('Training RecNet')
        RecNet.fit(
            x=x_train_obf,  # Input data (obfuscated images)
            y=x_train,  # Target data (original images)
            epochs=10,
            validation_data=(x_test_obf, x_test)
            # Validation data (used for monitoring model performance during training)
        )
        RecNet.save('models/mnist/recnet.h5')
    else:
        print('Trying to load RecNet weights')
        RecNet.load_weights('models/mnist/recnet.h5')

    # Try out the reconstruction model on the test data
    ind = [3, 2, 1, 18, 4, 8, 11, 0, 61, 7]
    recon = RecNet.predict(x_test_obf)
    h = []
    for i in ind:
        h.append(recon[i] * 255)
    v = np.concatenate(np.array(h), axis=1)
    cv2.imwrite("imgs/mnist/" + save_folder + '/reconstructed.jpg', v)

if __name__ == "__main__":
    for i in range(7):
        num_neurons = 2**(i+3)   # 8, 16, 32, 64, 128, 256, 512
        save_folder = str(num_neurons)
        print('\n\nObfNet neurons: ' + str(num_neurons) + '\n')

        main(
            # Control type of networks, use CNN or MLP
            is_inf_cnn=True,
            is_obf_cnn=True,
            is_rec_cnn=False,

            # Control the number of neurons in ObfNet and RecNet
            obf_num_neuron=num_neurons,
            rec_num_neuron=512,

            # If train is true, it will try to train the corresponding network on MNIST and save weights
            # Otherwise it will try to load the weights (will fail if not trained first, weights file won't be found)
            # Sometimes we want to try out different combinations of ObfNet and RecNet, without re-training the InfNet
            # Keep in mind that if the num_neurons is changed, the corresponding networks affected has to be re-trained
            train_inf=True,
            train_obf=True,
            train_rec=True,

            save_folder = save_folder
        )