from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import Utils
import PATHS
import tensorflow as tf


def evaluate_model(model, dir, pre_process=None):
    """
    Evaluates a model

    Parameters:
    - model (Tensorflow model): The model to be evaluated
    - dir (String): The directory to the dataset used to evaluate the model
    - pre_process (Function): An optional function to pre-process the input data from the directory

    Returns:
    float: (Loss, Accuracy)
    """
    test_datagen = ImageDataGenerator(preprocessing_function=pre_process)
    test_generator = test_datagen.flow_from_directory(
        directory=dir,
        target_size=(224, 224),  # Adjust size based on your model's input size
        batch_size=64,
        class_mode='categorical'  # Set to 'binary' if you have a binary classification problem
    )
    return model.evaluate(test_generator)

if __name__ == "__main__":
    model = Utils.get_pretrained_ResNet50(PATHS.RESNET_WEIGHTS)
    dir = PATHS.IMAGENET10c_PATH + 'val'
    model.compile(optimizer='adam',  # Use the optimizer of your choice
                  loss='categorical_crossentropy',  # Use the appropriate loss function for your problem
                  metrics=['accuracy'])  # Add any other metrics you want to track during evaluation

    evaluation = evaluate_model(model, dir, preprocess_input)
    print(f'Test Loss: {evaluation[0]}')
    print(f'Test Accuracy: {evaluation[1]}')