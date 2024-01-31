import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback, TensorBoard
from Helper import Logger
import PATHS
from Helper import Utils


def log_epoch(dir, epoch, logs):
    pass


class CustomCallback(Callback):
    def __init__(self, ts):
        super(CustomCallback, self).__init__()
        self.ts = ts
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        if self.ts.save:
            # Log the epoch for later plotting
            log_epoch(PATHS.HISTORY_DIR + self.ts.name + '/', epoch, logs)

            # Save weights and update info if they are new best
            current_val_loss = logs.get('val_loss')
            current_val_accuracy = logs.get('val_accuracy')
            if current_val_loss < self.best_val_loss and current_val_accuracy > self.best_val_accuracy:
                # Model's performance improved, save the model
                self.best_val_loss = current_val_loss
                self.best_val_accuracy = current_val_accuracy
                try:
                    obfnet, _ = Utils.split_models(self.model)
                    obfnet.save(PATHS.HISTORY_DIR + self.ts.name + '/best_obf_weights.h5')
                    self.ts.logger.new_best(epoch, current_val_accuracy, current_val_loss)
                except:
                    self.model.save(PATHS.HISTORY_DIR + self.ts.name + '/best_weights.h5')


def train(model, ts):
    # Create data generators with data augmentation for training set
    train_datagen = ImageDataGenerator(
        preprocessing_function=ts.preprocess_function
    )

    # Data generator for validation set (only rescaling)
    val_datagen = ImageDataGenerator(preprocessing_function=ts.preprocess_function)

    # Flow training images in batches using the generators
    train_generator = train_datagen.flow_from_directory(
        ts.train_dir,
        target_size=(ts.img_size, ts.img_size),
        batch_size=ts.batch_size,
        class_mode=ts.class_mode
    )

    # Flow validation images in batches using the generators
    val_generator = val_datagen.flow_from_directory(
        ts.val_dir,
        target_size=(ts.img_size, ts.img_size),
        batch_size=ts.batch_size,
        class_mode=ts.class_mode
    )

    # Compile the model
    model.compile(
        optimizer=ts.optimizer,
        loss=ts.loss,
        metrics=ts.metrics
    )

    tensorboard_callback = TensorBoard(log_dir=PATHS.HISTORY_DIR + ts.name + '/TensorBoard', histogram_freq=1)
    custom_callback = CustomCallback(ts)
    cb = [custom_callback]
    if ts.save:
        logger = Logger.Logger(ts, model.name)
        ts.logger = logger
        cb.append(tensorboard_callback)

    # Train the model
    history = model.fit(
        train_generator,
        epochs=ts.epochs,
        validation_data=val_generator,
        callbacks=cb
    )

    return history
