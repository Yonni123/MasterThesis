import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(1)
np.random.seed(1)


def add_random_color(image):
    # Choose a random channel to keep and remove the rest
    keep_channel = np.random.randint(0, 7)
    z = np.zeros_like(image)

    colored = None
    if keep_channel == 0:  # There's probably easier and faster way to do this, im just too lazy
        colored = np.stack([image, z, z], axis=-1)
    elif keep_channel == 1:
        colored = np.stack([z, image, z], axis=-1)
    elif keep_channel == 2:
        colored = np.stack([z, z, image], axis=-1)
    elif keep_channel == 3:
        colored = np.stack([z, image, image], axis=-1)
    elif keep_channel == 4:
        colored = np.stack([image, z, image], axis=-1)
    elif keep_channel == 5:
        colored = np.stack([image, image, z], axis=-1)
    elif keep_channel == 6:
        colored = np.stack([image, image, image], axis=-1)
    return colored.astype(np.uint8)


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    print("Total training data: " + str(num_train_samples))
    print("Total testing data: " + str(num_test_samples))

    print("\nColorizing all images")
    x_train_colored = np.zeros((num_train_samples, 28, 28, 3))
    x_test_colored = np.zeros((num_test_samples, 28, 28, 3))
    for i in range(len(x_train)):
        x_train_colored[i] = add_random_color(x_train[i])
    for i in range(len(x_test)):
        x_test_colored[i] = add_random_color(x_test[i])

    X_train_obf, X_train_rec, y_train_obf, y_train_rec = train_test_split(x_train_colored, y_train, test_size=0.5,
                                                                          random_state=1)
    X_test_obf, X_test_rec, y_test_obf, y_test_rec = train_test_split(x_test_colored, y_test, test_size=0.5,
                                                                      random_state=1)
    print("\nTotal training data for ObfNet: " + str(len(X_train_obf)))
    print("Total testing data for ObfNet: " + str(len(X_test_obf)))
    print("\nTotal training data for RecNet: " + str(len(X_train_rec)))
    print("Total testing data for RecNet: " + str(len(X_test_rec)))

    # Display 3x5 grid of original and colored images side by side
    num_rows = 3
    num_cols = 5
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(2 * num_cols, num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            # Display original images
            axes[i, 2 * j].imshow(x_train[i * num_cols + j], cmap='gray')
            axes[i, 2 * j].axis('off')

            # Display colored images (normalize pixel values to [0, 1])
            axes[i, 2 * j + 1].imshow(x_train_colored[i * num_cols + j] / 255.0)
            axes[i, 2 * j + 1].axis('off')

    plt.show()

    return {
    "Obf": [X_train_obf, X_test_obf, y_train_obf, y_test_obf],
    "Rec": [X_train_rec, X_test_rec, y_train_rec, y_test_rec]
    }
