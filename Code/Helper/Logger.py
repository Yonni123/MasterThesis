import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.resnet50 as resnet50
import os

import Helper
import PATHS


def dump_to_txt(array, filename):
    # Open the file in 'w' mode to write (and create if it doesn't exist)
    # If the file already exists, it will overwrite its content
    with open(filename, 'w') as file:
        # Write each element in the array as a new line in the file
        for element in array:
            file.write(element + '\n')

class Logger:
    def __init__(self, ts, model_name):
        self.ts = ts
        self.model_name = model_name
        self.session_path = PATHS.HISTORY_DIR + ts.name + '/'
        os.makedirs(PATHS.HISTORY_DIR + ts.name, exist_ok=True)
        with open(self.session_path + '/info.txt', 'w') as txt_file:
            txt_file.write('ObfNet name: ' + model_name)

    def new_best(self, epoch, acc, loss):
        epoch_txt = f'Best epoch:\t {epoch + 1}'
        acc_txt = f'Best acc:\t {acc*100:.2f}%'
        loss_txt = f'Best loss:\t {loss}'
        dump = ['ObfNet name: ' + self.model_name, epoch_txt, acc_txt, loss_txt]
        dump_to_txt(dump, self.session_path + 'info.txt')
