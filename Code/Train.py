import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.applications.resnet50 as resnet50
import os
import numpy as np
import random

record = keras.callbacks.History()

import time
import Helper
import TrainingSettings

class CustomSaver(keras.callbacks.Callback):
    def __init__(self, path, freq):
        self.path = path
        self.freq = freq

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        global record
        if self.freq==0:
            return
        if epoch>1 and not(epoch % self.freq):  # save each k-th epoch etc.
            Helper.savetofile(self.path+".hist", record.history)
            print(record)

class CheckPoint(tf.keras.callbacks.ModelCheckpoint):
    def __init__(self, filepath, monitor, verbose, save_best_only,
                 save_weights_only, save_freq):
        self.freq = save_freq
        self.path = filepath
        filepath += "best.h5"
        super().__init__(filepath=filepath,
                         monitor=monitor,
                         verbose=verbose,
                         save_best_only=save_best_only,
                         save_weights_only=save_weights_only,
                         save_freq='epoch')

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        global record
        if self.freq==0:
            return
        if epoch>1 and not(epoch % self.freq):
            Helper.savetofile(self.path+"data.hist", record.history)
            super().on_epoch_end(epoch, logs)

def loadData(c):
    n_train = sum([len(files) for _, _, files in os.walk(c.dTranDir)])
    n_val = sum([len(files) for _, _, files in os.walk(c.dValDir)])
    print("Number of training samples : "+str(n_train))
    print("Number of validation samples : {}"+str(n_val))

    # Set PIL to allow image files that are truncated.
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=resnet50.preprocess_input,
        rotation_range=c.rotation_range,
        brightness_range=c.brightness_range,
        width_shift_range=c.width_shift_range,
        height_shift_range=c.height_shift_range,
        zoom_range=c.zoom_range,
        channel_shift_range=c.channel_shift_range,
        data_format=K.image_data_format()
    )

    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=resnet50.preprocess_input,
        data_format=K.image_data_format()
    )

    tg = train_datagen.flow_from_directory(
        c.dTranDir,
        target_size=(c.imgSize, c.imgSize),
        batch_size=c.batch_size,
        shuffle=True,
        class_mode='categorical'
    )

    vg = val_datagen.flow_from_directory(
        c.dValDir,
        target_size=(c.imgSize, c.imgSize),
        batch_size=c.batch_size,
        shuffle=False,
        class_mode='categorical'
    )

    return (tg, vg)

class HackyMcHackface(tf.keras.utils.Sequence):
    def __init__(self, it):
        self.it = it

    def __len__(self):
        return self.it.__len__()

    def __getitem__(self, idx):
        (a,b) = self.it.__getitem__(idx)
        return (a[:,:,:,:3],b)

    def on_epoch_end(self):
        return self.it.on_epoch_end()

def train(model, data, c):
    (tg, vg) = data
    global record
    callbacks = [
        record,
        #CustomSaver(name, savefreq),
        CheckPoint(
            filepath=c.path,
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq=c.savefreq
        )
    ]
    if (c.earltstop):
        callbacks.append(tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            min_delta=c.min_delta,
            patience=c.patience,
            mode="min",
            restore_best_weights=True,
        ))
    if(c.startfrom==False):
        model.compile(optimizer=tf.keras.optimizers.Adam(c.trainingRate), loss=keras.losses.CategoricalCrossentropy(label_smoothing=c.label_smoothing), metrics=["accuracy"])
    else:
        keras.backend.set_value(model.optimizer.learning_rate, c.trainingRate)
        #keras.backend.set_value(model.loss,keras.losses.CategoricalCrossentropy(label_smoothing=c.label_smoothing))
    try:
        model.fit(tg, epochs=c.epochs , callbacks=callbacks, validation_data=vg, class_weight=c.class_weight)
    except KeyboardInterrupt:
        print("Aborted training!")
    return record