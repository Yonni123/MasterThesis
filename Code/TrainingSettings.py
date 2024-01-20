import Helper
import Logger

class TrainingSettings:
    def __init__(self):
        # data augmentation (turned off by default)
        self.rotation_range = 0         # degrees (0 to 180)
        self.brightness_range = [1, 1]  # fraction or tuple of fraction (min, max). 1.0 is original brightness
        self.width_shift_range = 0      # fraction of total width
        self.height_shift_range = 0     # fraction of total height
        self.zoom_range = [1, 1]        # zoom range
        self.channel_shift_range = 0    # fraction of total channels

        # data directory
        self.train_dir = ''   # path to training data
        self.val_dir = ''    # path to validation data

        # training settings
        self.save = True            # save model and logs
        self.save_freq = 10         # save model every n epochs
        self.save_dir = 'models/'   # path to save model and logs
        self.early_stop = True      # stop training early if validation loss does not improve
        self.patience = 4           # number of epochs to wait before early stop
        self.min_delta = 0          # minimum change in validation loss to be considered as improvement
        self.imgSize = 224          # image size
        self.batch_size = 32        # batch size
        self.epochs = 500           # number of epochs
        self.learning_rate = 1e-5   # learning rate
        self.label_smoothing = 0    # label smoothing
        self.dropout = 0            # dropout rate
        self.classes = 1000         # number of classes

        # ObfNet shape  input -> (layer, layer) -> output, layer = [type, size]
        # type = 0: Dense, 1: Conv2D, size = number of neurons or filters
        self.obf_shape = ([0, 1000], [0, 1000])