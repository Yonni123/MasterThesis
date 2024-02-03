# This is the original source code for ObfNet taken from:
# https://github.com/ntu-aiot/ObfNet/blob/master/asl/main.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, Dense, Dropout, Flatten, Reshape, Activation, MaxPooling2D
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pickle as pkl
from PIL import Image
import datetime
import tensorflow as tf
# set random seed
np.random.seed(1)
tf.random.set_seed(1)


def get_data(folder):
    """Load the data and labels from the given folder."""
    train_len = 87000
    imageSize = 64

    X = np.empty((train_len, imageSize, imageSize, 3), dtype=np.float32)
    y = np.empty((train_len, ), dtype=int)
    cnt = 0

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            if folderName in ['A']:
                label = 0
            elif folderName in ['B']:
                label = 1
            elif folderName in ['C']:
                label = 2
            elif folderName in ['D']:
                label = 3
            elif folderName in ['E']:
                label = 4
            elif folderName in ['F']:
                label = 5
            elif folderName in ['G']:
                label = 6
            elif folderName in ['H']:
                label = 7
            elif folderName in ['I']:
                label = 8
            elif folderName in ['J']:
                label = 9
            elif folderName in ['K']:
                label = 10
            elif folderName in ['L']:
                label = 11
            elif folderName in ['M']:
                label = 12
            elif folderName in ['N']:
                label = 13
            elif folderName in ['O']:
                label = 14
            elif folderName in ['P']:
                label = 15
            elif folderName in ['Q']:
                label = 16
            elif folderName in ['R']:
                label = 17
            elif folderName in ['S']:
                label = 18
            elif folderName in ['T']:
                label = 19
            elif folderName in ['U']:
                label = 20
            elif folderName in ['V']:
                label = 21
            elif folderName in ['W']:
                label = 22
            elif folderName in ['X']:
                label = 23
            elif folderName in ['Y']:
                label = 24
            elif folderName in ['Z']:
                label = 25
            elif folderName in ['del']:
                label = 26
            elif folderName in ['nothing']:
                label = 27
            elif folderName in ['space']:
                label = 28
            else:
                label = 29
            for image_filename in os.listdir(folder + '\\' + folderName):
                img_file = Image.open(folder + '\\' + folderName + '\\' +
                                      image_filename)
                # img_file = cv2.imread(folder + folderName + '/' + image_filename)
                if img_file is not None:
                    # img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    # img_arr = np.asarray(img_file).reshape((-1, imageSize, imageSize, 3))
                    img_file.thumbnail((imageSize, imageSize), Image.ANTIALIAS)
                    img_arr = np.array(img_file).reshape(
                        (-1, imageSize, imageSize, 3))

                    X[cnt] = img_arr
                    y[cnt] = label
                    cnt += 1

    with open("train.pkl", "wb") as f:
        pkl.dump((X, y), f)

    return X, y


def grayscale(data, dtype='float32'):
    """Convert colored image data to greyscale."""

    r, g, b = np.asarray(.3, dtype=dtype), np.asarray(
        .59, dtype=dtype), np.asarray(.11, dtype=dtype)
    rst = r * data[:, :, :, 0] + g * data[:, :, :, 1] + b * data[:, :, :, 2]
    rst = np.expand_dims(rst, axis=3)
    return rst


def get_inference_model_mlp(target_dims, num_classes, num_layers=6):
    model = Sequential()
    model.add(Flatten(input_shape=(target_dims)))
    for i in range(num_layers):
        model.add(Dense(1024))
        model.add(Activation('relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_inference_model_cnn(target_dims, num_classes):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=(target_dims)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def get_obfmodel_mlp(target_dims, num_neuron=512):
    model = Sequential()
    model.add(Flatten(input_shape=(target_dims)))
    model.add(Dense(num_neuron))
    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model


def get_obfmodel_cnn(target_dims, num_neuron):
    model = Sequential()
    model.add(
        Conv2D(32,
               kernel_size=(3, 3),
               activation='relu',
               input_shape=(target_dims)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(num_neuron))
    model.add(Activation('relu'))
    model.add(Dense(np.prod(target_dims)))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Reshape(target_dims))
    return model


def main(is_inf_cnn, is_obf_cnn, is_rec_cnn, obf_num_neuron, rec_num_neuron, train_obf, train_inf, train_rec, save_folder, dataset_path):

    os.makedirs("models/asl/", exist_ok=True)

    batch_size = 64
    imageSize = 64
    train_len = 87000
    target_dims = (imageSize, imageSize, 3)
    num_classes = 29
    max_epochs = 100

    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

    with open("train.pkl", "rb") as f:
        X_train, y_train = pkl.load(f)

    X_train, X_test, y_train, y_test = train_test_split(X_train,
                                                        y_train,
                                                        test_size=0.1)

    # Encode labels to hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
    y_trainHot = to_categorical(y_train, num_classes=num_classes)
    y_testHot = to_categorical(y_test, num_classes=num_classes)

    if train_inf:
        if is_inf_cnn:
            inference_model = get_inference_model_cnn(target_dims, num_classes)
            inf_path = 'models/asl/inf-cnn.h5'
        else:
            inference_model = get_inference_model_mlp(target_dims, num_classes)
            inf_path = 'models/asl/inf-mlp.h5'

        inference_model.compile(optimizer='adam',
                                loss='categorical_crossentropy',
                                metrics=["accuracy"])

        infmodelcheck_cb = ModelCheckpoint(inf_path,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='max')

        inference_model.fit(x=X_train,
                            y=y_trainHot,
                            batch_size=batch_size,
                            epochs=max_epochs,
                            callbacks=[infmodelcheck_cb, tensorboard_cb],
                            validation_data=(X_test, y_testHot))
    else:
        inference_model = load_model('models/asl/inf.h5')


    print(inference_model.summary())
    print('Inference Result: ')
    print(inference_model.evaluate(X_test, y_testHot, verbose=0))
    quit()
    ######################
    # training of ObfNet #
    ######################

    if is_obf_cnn:
        obfmodel = get_obfmodel_cnn(target_dims, obf_num_neuron)
    else:
        obfmodel = get_obfmodel_mlp(target_dims, obf_num_neuron)
        obfmodel.build((None, ) + target_dims)

    inference_model.trainable = False
    for l in inference_model.layers:
        l.trainable = False

    combined_model = Model(inputs=obfmodel.input,
                           outputs=inference_model(obfmodel.output))
    combined_model.compile(optimizer='adamdelta',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    print(combined_model.summary())
    com_path = 'models/asl/combined-model.h5'

    combined_model.fit(x=X_train,
                       y=y_trainHot,
                       batch_size=batch_size,
                       epochs=max_epochs,
                       callbacks=[
                           ModelCheckpoint(com_path,
                                           monitor='val_accuracy',
                                           verbose=1,
                                           save_best_only=True,
                                           mode='max')
                       ],
                       verbose=1,
                       validation_data=(X_test, y_testHot))

    ######################
    # testing of ObfNet #
    ######################

    score = combined_model.evaluate(X_test, y_testHot, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    pos = {}
    for i in range(num_classes):
        pos[i] = -1

    for ind, num in enumerate(y_test):
        pos[num] = ind
        if all(x != -1 for x in pos.values()):
            break

    cls_pos = list(x for x in pos.values())

    # from PIL import Image
    # import cv2
    #
    # def get_concat_h(im1, im2):
    #     dst =  Image.new('RGB', (im1.width + im2.width, im1.height))
    #     dst.paste(im1, (0, 0))
    #     dst.paste(im2, (im1.width, 0))
    #     return dst
    #
    # h1 = []
    # h2 = []
    #
    # for i in range(0, 11):
    #     # h1 = Image.fromarray(X_test[i], 'RGB')
    #     # h2 = Image.fromarray(layer_output[i], 'RGB')
    #     # get_concat_h(h1, h2).save('imgs/c'+str(i)+'.jpg')
    #     h1.append(X_test[i])
    #     h2.append(layer_output[i])
    #
    # v1 = np.concatenate(np.array(h1), axis=1)
    # v2 = np.concatenate(np.array(h2), axis=1)
    # v = np.concatenate((v1, v2), axis=0)
    # cv2.imwrite('imgs/cat.jpg', v)
    # cv2.imwrite('imgs/cat1.jpg', v1)
    # cv2.imwrite('imgs/cat2.jpg', v2)


if __name__ == "__main__":
    main(
        # Control type of networks, use CNN or MLP
        is_inf_cnn=False,
        is_obf_cnn=True,
        is_rec_cnn=False,

        # Control the number of neurons in ObfNet and RecNet
        obf_num_neuron=128,
        rec_num_neuron=128,

        # If train is true, it will try to train the corresponding network on MNIST and save weights
        # Otherwise it will try to load the weights (will fail if not trained first, weights file won't be found)
        # Sometimes we want to try out different combinations of ObfNet and RecNet, without re-training the InfNet
        # Keep in mind that if the num_neurons is changed, the corresponding networks affected has to be re-trained
        train_inf=False,
        train_obf=True,
        train_rec=True,

        save_folder='a',
        dataset_path='C:\ASL\\asl_alphabet_train'
    )