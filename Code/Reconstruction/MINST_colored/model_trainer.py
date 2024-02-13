from keras import Sequential
from keras.layers import *
from keras.utils import to_categorical

from colored_MNIST_generator import generate_data


def preprocess_data(data):
    x_train, x_test, y_train, y_test = data

    # Normalize data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # convert class vectors to binary class matrices
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, x_test, y_train, y_test


def get_inference_cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2), padding='same'))
    model.add(Flatten())
    model.add(Dense(num_classes))
    model.add(Softmax())
    return model


def get_obfmodel_mlp(input_shape, num_neuron=100):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(num_neuron, activation='relu'))
    model.add(Dense(28 * 28, activation='relu'))
    model.add(Reshape(input_shape))
    return model


def train_inf(x_train, x_test, y_train, y_test):
    model = get_inference_mlp(
        input_shape=(28, 28, 3),
        num_classes=10
    )
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=10,
        validation_data=(x_test, y_test)
    )


