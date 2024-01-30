import datetime


def generate_session_name():
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{current_time}"
    return session_name


class TrainingSettings:
    def __init__(self):
        # name of current training session
        self.name = generate_session_name()

        # data directory
        self.train_dir = ''  # path to training data
        self.val_dir = ''  # path to validation data
        self.input_shape = (224, 224, 3)  # input shape, (height, width, channels)
        self.class_mode = 'categorical'  # or 'binary' depending on your problem
        self.preprocess_function = None

        # logs
        self.save = True  # save model and logs
        self.save_freq = 10  # save model every n epochs
        self.save_dir = '../History/'  # path to save model and logs

        # training settings
        self.early_stop = True  # stop training early if validation loss does not improve
        self.patience = 4  # number of epochs to wait before early stop
        self.min_delta = 0  # minimum change in validation loss to be considered as improvement
        self.img_size = 224  # image size
        self.batch_size = 32  # batch size
        self.epochs = 50  # number of epochs
        self.learning_rate = 1e-5  # learning rate
        self.classes = 1000  # number of classes

        # Optimizer and loss
        self.optimizer = 'adam'
        self.loss = 'categorical_crossentropy',  # or 'binary_cross entropy'
        self.metrics = ['accuracy']
