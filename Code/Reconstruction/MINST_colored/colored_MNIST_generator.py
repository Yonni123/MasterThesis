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
    return colored.astype(np.uint8), keep_channel


def generate_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    num_train_samples = x_train.shape[0]
    num_test_samples = x_test.shape[0]
    print("Total training data: " + str(num_train_samples))
    print("Total testing data: " + str(num_test_samples))

    print("\nColorizing all images")
    x_train_colored = np.zeros((num_train_samples, 28, 28, 3))
    x_test_colored = np.zeros((num_test_samples, 28, 28, 3))
    train_colors = np.zeros(num_train_samples)
    test_colors = np.zeros(num_test_samples)
    for i in range(len(x_train)):
        x_train_colored[i], train_colors[i] = add_random_color(x_train[i])
    for i in range(len(x_test)):
        x_test_colored[i], test_colors[i] = add_random_color(x_test[i])

    stacked_train_labels = np.vstack((y_train, train_colors)).T
    stacked_test_labels = np.vstack((y_test, test_colors)).T

    X_train_obf, X_train_rec, stacked_train_obf, stacked_train_rec = train_test_split(x_train_colored, stacked_train_labels, test_size=0.5,
                                                                          random_state=1)
    X_test_obf, X_test_rec, stacked_test_obf, stacked_test_rec = train_test_split(x_test_colored, stacked_test_labels, test_size=0.5,
                                                                      random_state=1)
    print("\nTotal training data for ObfNet: " + str(len(X_train_obf)))
    print("Total testing data for ObfNet: " + str(len(X_test_obf)))
    print("\nTotal training data for RecNet: " + str(len(X_train_rec)))
    print("Total testing data for RecNet: " + str(len(X_test_rec)))

    # We need to split the digit information from the color information
    y_train_obf, c_train_obf = stacked_train_obf[:, 0], stacked_train_obf[:, 1]
    y_test_obf, c_test_obf = stacked_test_obf[:, 0], stacked_test_obf[:, 1]
    y_train_rec, c_train_rec = stacked_train_rec[:, 0], stacked_train_rec[:, 1]
    y_test_rec, c_test_rec = stacked_test_rec[:, 0], stacked_test_rec[:, 1]

    # Display 3x5 grid of original and colored images side by side
    num_rows = 4
    num_cols = 5
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(3 * num_cols, 2*num_rows))
    color_index = ['Red', 'Green', 'Blue', 'Aqua', 'Magenta', 'Yellow', 'White']

    for i in range(num_rows):
        for j in range(num_cols):
            # Display original images
            axes[i, 2 * j].imshow(x_train[i * num_cols + j], cmap='gray')
            axes[i, 2 * j].axis('off')
            axes[i, 2 * j].set_title('ORIGINAL')

            # Display colored images (normalize pixel values to [0, 1])
            axes[i, 2 * j + 1].imshow(x_train_colored[i * num_cols + j] / 255.0)
            axes[i, 2 * j + 1].axis('off')
            axes[i, 2 * j + 1].set_title(color_index[int(train_colors[i * num_cols + j])])
    plt.show()

    return {
    "Obf": [X_train_obf, X_test_obf, y_train_obf, y_test_obf, c_train_obf, c_test_obf],
    "Rec": [X_train_rec, X_test_rec, y_train_rec, y_test_rec, c_train_rec, c_test_rec]
    }


def generate_solid_data(N=30000):
    unique_colors = np.random.rand(N, 1, 1, 3)
    random_colors = unique_colors * np.ones((N, 28, 28, 3), dtype=np.uint8)
    # Display 3x5 grid of original and colored images side by side
    num_rows = 4
    num_cols = 5
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(3 * num_cols, 2 * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            # Display original images
            axes[i, 2 * j].imshow(random_colors[i * num_cols + j])
            axes[i, 2 * j].axis('off')

            # Display colored images (normalize pixel values to [0, 1])
            axes[i, 2 * j + 1].imshow(random_colors[N-1-(i * num_cols + j)])
            axes[i, 2 * j + 1].axis('off')
    plt.show()
    return random_colors

def generate_noisy_data(N=30000):
    # Image dimensions
    width, height, channels = 28, 28, 1
    # Generate static noise in the middle of the image
    middle_x, middle_y = width // 2, height // 2
    noise_intensity = 255  # Maximum intensity for white noise

    colored_images = np.zeros(shape=(N, 28, 28, 3))
    classes = np.zeros(N)

    for n in range(N):
        if n % 100 == 0 and n != 0:
            print(f"Generated {n} out of {N} images...")

        image = np.zeros((height, width, channels), dtype=np.uint8)
        for y in range(height):
            for x in range(width):
                distance_to_center = np.sqrt((x - middle_x) ** 2 + (y - middle_y) ** 2)
                intensity = int(np.random.rand() * noise_intensity * (1 - distance_to_center / (width / 2)))
                image[y, x, 0] = np.clip(intensity, 0, 255)

        # Add random color to the image
        image = np.squeeze(image, axis=2)
        colored_image, keep_channel = add_random_color(image)
        colored_image = colored_image.astype(np.float32)/noise_intensity
        colored_images[n] = colored_image
        classes[n] = keep_channel
    print(f"Generated {N} out of {N} images...")

    # Display 3x5 grid of original and colored images side by side
    color_index = ['Red', 'Green', 'Blue', 'Aqua', 'Magenta', 'Yellow', 'White']
    num_rows = 4
    num_cols = 5
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(3 * num_cols, 2 * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            # Display original images
            axes[i, 2 * j].imshow(colored_images[i * num_cols + j])
            axes[i, 2 * j].axis('off')
            axes[i, 2 * j].set_title(color_index[int(classes[i * num_cols + j])])

            # Display colored images (normalize pixel values to [0, 1])
            axes[i, 2 * j + 1].imshow(colored_images[N - 1 - (i * num_cols + j)])
            axes[i, 2 * j + 1].axis('off')
            axes[i, 2 * j + 1].set_title(color_index[int(classes[N - 1 - (i * num_cols + j)])])
    plt.show()
    return colored_images, classes


def generate_path_data(N=30000):
    # Image dimensions
    width, height, channels = 28, 28, 1
    # Generate static noise in the middle of the image
    middle_x, middle_y = width // 2, height // 2
    noise_intensity = 255  # Maximum intensity for white noise
    step_size = 2  # Adjust the step size
    thickness = 1

    colored_images = np.zeros(shape=(N, 28, 28, 3))
    classes = np.zeros(N)

    for n in range(N):
        if n % 100 == 0 and n != 0:
            print(f"Generated {n} out of {N} images...")

        image = np.zeros((height, width, channels), dtype=np.uint8)

        current_position = np.array([width // 2, height // 2])  # Start at the center of the image

        # Randomly generate a path for the digit
        num_steps = 30
        path = np.cumsum(np.random.randint(-step_size, step_size + 1, size=(num_steps, 2)), axis=0)

        for step in path:
            x, y = current_position + step
            x, y = int(x), int(y)

            if 0 <= x < width and 0 <= y < height:
                for i in range(-thickness // 2, thickness // 2 + 1):
                    for j in range(-thickness // 2, thickness // 2 + 1):
                        nx, ny = x + i, y + j
                        if 0 <= nx < width and 0 <= ny < height:
                            intensity = int(np.random.normal(loc=noise_intensity, scale=noise_intensity / 2))
                            intensity = np.clip(intensity, 0, 255)
                            image[ny, nx, 0] = intensity

        # Add random color to the image
        image = np.squeeze(image, axis=2)
        colored_image, keep_channel = add_random_color(image)
        colored_image = colored_image.astype(np.float32)/noise_intensity
        colored_images[n] = colored_image
        classes[n] = keep_channel
    print(f"Generated {N} out of {N} images...")

    # Display 3x5 grid of original and colored images side by side
    color_index = ['Red', 'Green', 'Blue', 'Aqua', 'Magenta', 'Yellow', 'White']
    num_rows = 4
    num_cols = 5
    fig, axes = plt.subplots(num_rows, 2 * num_cols, figsize=(3 * num_cols, 2 * num_rows))

    for i in range(num_rows):
        for j in range(num_cols):
            # Display original images
            axes[i, 2 * j].imshow(colored_images[i * num_cols + j])
            axes[i, 2 * j].axis('off')
            axes[i, 2 * j].set_title(color_index[int(classes[i * num_cols + j])])

            # Display colored images (normalize pixel values to [0, 1])
            axes[i, 2 * j + 1].imshow(colored_images[N - 1 - (i * num_cols + j)])
            axes[i, 2 * j + 1].axis('off')
            axes[i, 2 * j + 1].set_title(color_index[int(classes[N - 1 - (i * num_cols + j)])])
    plt.show()
    return colored_images, classes

if __name__ == '__main__':
    data, c = generate_path_data(N=50)
    print('a')