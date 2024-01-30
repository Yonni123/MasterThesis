import Train
import TrainingSettings as TS
from Helper import Utils
import Helper
import PATHS
import ObfNet_Desgins as OND
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input


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

    # Train the model
    history = model.fit(
        train_generator,
        epochs=ts.epochs,
        validation_data=val_generator
    )

    # Save the trained model
    if ts.save:
        model.save(ts.name + '_final.h5')

    return history
