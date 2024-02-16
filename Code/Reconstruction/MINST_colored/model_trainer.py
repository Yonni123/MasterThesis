import numpy as np
from keras import Sequential
from keras.layers import *
from keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model


def preprocess_data(data):
    x_train, x_test, y_train, y_test, c_train, c_test = data

    # Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    c_train = to_categorical(c_train, 7)
    c_test = to_categorical(c_test, 7)

    return x_train, x_test, y_train, y_test, c_train, c_test


def get_color_model_mlp(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model


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


def get_obfmodel_mlp(input_shape, num_neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(np.prod(input_shape), activation='relu'))
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
    model.add(Dense(np.prod(input_shape), activation='relu'))
    model.add(Reshape(input_shape))
    return model


def train_inf_model(x_train, x_test, y_train, y_test):
    model = get_inference_cnn(
        input_shape=(28, 28, 3),
        num_classes=10
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[
            ModelCheckpoint('inf_model.h5',
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
        ]
    )
    return load_model('inf_model.h5')   # Return the best model


def join_models(obfuscation_model, inference_model):
    """
    Join an obfuscation model and an inference model into a combined model.

    Parameters:
    - obfuscation_model (Sequential): The obfuscation model.
    - inference_model (Sequential): The inference model.

    Returns:
    Sequential: The combined model.

    Raises:
    Exception: If the input and output shapes of the obfuscation model are not equal to the input shape of the inference model.
    """
    # Make sure that the input of the obfuscation model is equal to the output of the inference model
    if obfuscation_model.input_shape != inference_model.input_shape or obfuscation_model.output_shape != inference_model.input_shape:
        raise Exception('The input and output shapes of the obfuscation model are not equal to the input shape of the inference model')

    obfnet_name = obfuscation_model.name
    combined_model = Sequential(name=obfnet_name)
    obfuscation_model._name = 'ObfModel'
    inference_model._name = 'InfModel'
    combined_model.add(obfuscation_model)
    combined_model.add(inference_model)
    return combined_model


def split_models(combined_model):
    """
    Split a combined model into its obfuscation model and inference model components.

    Parameters:
    - combined_model (Sequential): The combined model containing both obfuscation and inference models.

    Returns:
    tuple: A tuple containing the obfuscation model and the inference model.

    Raises:
    Exception: If the number of layers in the combined model is not 2.
    Exception: If 'ObfModel' is not found as the first layer in the combined model.
    Exception: If 'InfModel' is not found as the second layer in the combined model.

    Note:
    This function only works with combined models generated by join_models() function
    """
    if len(combined_model.layers) != 2:
        raise Exception('Wrong layer size, must be 2')
    if combined_model.layers[0].name != 'ObfModel':
        raise Exception('ObfModel was not found in the combined model')
    if combined_model.layers[1].name != 'InfModel':
        raise Exception('InfModel was not found in the combined model')
    return combined_model.layers[0], combined_model.layers[1]


def train_obf_model_cnn(x_train, x_test, y_train, y_test, inf_model, bottleneck=512):
    obf_model = get_obfmodel_cnn(
        input_shape=(28, 28, 3),
        num_neuron=bottleneck
    )
    combined_model = join_models(obf_model, inf_model)  # Join Obf and Inf together into a combined model
    combined_model.layers[1].trainable = False          # Freeze the second layer, which is the inference model
    combined_model.summary()

    combined_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    combined_model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[
            ModelCheckpoint(f'ObfNet_cnn/combined_model_{bottleneck}.h5',
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
        ]
    )
    combined_model = load_model(f'ObfNet_cnn/combined_model_{bottleneck}.h5')
    obf, inf = split_models(combined_model)
    return obf  # We don't need the inference network


def train_obf_model_mlp(x_train, x_test, y_train, y_test, inf_model, bottleneck=512):
    obf_model = get_obfmodel_mlp(
        input_shape=(28, 28, 3),
        num_neuron=bottleneck
    )
    combined_model = join_models(obf_model, inf_model)  # Join Obf and Inf together into a combined model
    combined_model.layers[1].trainable = False          # Freeze the second layer, which is the inference model
    combined_model.summary()

    combined_model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    combined_model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=30,
        validation_data=(x_test, y_test),
        callbacks=[
            ModelCheckpoint(f'ObfNet_mlp/combined_model_{bottleneck}.h5',
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
        ]
    )
    combined_model = load_model(f'ObfNet_mlp/combined_model_{bottleneck}.h5')
    obf, inf = split_models(combined_model)
    return obf  # We don't need the inference network


def train_rec_model_mlp(ground_truth_train_images, ground_truth_test_images, obf_model, save_path='best_weights.h5'):
    # Generate x_train and x_test by obfuscating the ground truth images
    x_train = obf_model.predict(ground_truth_train_images)
    x_test = obf_model.predict(ground_truth_test_images)
    y_train = ground_truth_train_images # Just for more convenient naming
    y_test = ground_truth_test_images

    # Load a model architecture for ObfNet, since this is MLP, we will use ObfNet architecture with largest bottleneck
    RecNet = get_obfmodel_mlp((28, 28, 3), 1024)
    RecNet.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
    RecNet.fit(
        x=x_train,  # Input data (obfuscated images)
        y=y_train,  # Target data (original images)
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks = [
            ModelCheckpoint(save_path,
                        monitor='val_accuracy',
                        verbose=1,
                        save_best_only=True,
                        mode='max')
        ]
    )
    return load_model(save_path)


def train_color_model_mlp(ground_truth_train_images, ground_truth_test_images, c_train, c_test, obf_model, save_path='best_weights.h5'):
    # Generate x_train and x_test by obfuscating the ground truth images
    x_train = obf_model.predict(ground_truth_train_images)
    x_test = obf_model.predict(ground_truth_test_images)

    # Load a model architecture for ObfNet, since this is MLP, we will use ObfNet architecture with largest bottleneck
    ColorNet = get_color_model_mlp((28, 28, 3), 7)  # We have seven colors
    ColorNet.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    ColorNet.fit(
        x_train, c_train,
        batch_size=64,
        epochs=50,
        validation_data=(x_test, c_test),
        callbacks=[
            ModelCheckpoint(save_path,
                            monitor='val_accuracy',
                            verbose=1,
                            save_best_only=True,
                            mode='max')
        ]
    )
    return load_model(save_path)  # Return the best model