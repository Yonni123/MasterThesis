import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.models import load_model
from data_generator import generate_colored_MNIST
from model_trainer import *
from RecNet_visualizer import visualize_recnets
import os

from sklearn.decomposition import PCA

def PCA_analysis(data, labels, n_components=2, blocking=True, title='PCA Scatter Plot'):
    # Reshape the images into 2D arrays
    flattened = data.reshape(data.shape[0], -1)

    # Normalize the data (optional but recommended)
    td_normalized = flattened / 255.0  # Assuming pixel values are in the range [0, 255]

    # Instantiate PCA with the desired number of components
    pca = PCA(n_components=n_components, random_state=42)

    # Fit PCA on the normalized data
    pca_result = pca.fit_transform(td_normalized)

    if len(np.unique(labels)) == 7:
        # Define colors for each class
        class_colors = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'aqua',
            4: 'magenta',
            5: 'yellow',
            6: 'black'
        }

        # Create a scatter plot with specified colors for each class
        plt.figure(figsize=(10, 8))
        color_index = ['Red', 'Green', 'Blue', 'Aqua', 'Magenta', 'Yellow', 'White']

        for class_label, color in class_colors.items():
            indices = np.where(labels == class_label)
            plt.scatter(pca_result[indices, 0], pca_result[indices, 1],
                        label=f'{color_index[int(class_label)]} digits', color=color)

    else:
        # Define colors for each class
        class_colors = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'aqua',
            4: 'magenta',
            5: 'yellow',
            6: 'black',
            7: 'brown',
            8: 'purple',
            9: 'orange'
        }

        # Create a scatter plot with specified colors for each class
        plt.figure(figsize=(10, 8))

        for class_label, color in class_colors.items():
            indices = np.where(labels == class_label)
            plt.scatter(pca_result[indices, 0], pca_result[indices, 1],
                        label=f'{int(class_label)} digits', color=color)

    plt.title(title)
    plt.xlabel('PCA Dimension 1')
    plt.ylabel('PCA Dimension 2')
    plt.legend()
    plt.show(block=blocking)

def tSNE_analysis(data, labels, perplexity=30, blocking=True, title='t-SNE Scatter Plot'):
    # Reshape the images into 2D arrays
    flattened = data.reshape(data.shape[0], -1)

    # Normalize the data (optional but recommended)
    td_normalized = flattened / 255.0  # Assuming pixel values are in the range [0, 255]

    # Instantiate t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)  # You can adjust the perplexity as needed

    # Fit t-SNE on the normalized data
    tsne_result = tsne.fit_transform(td_normalized)

    if len(np.unique(labels)) == 7:
        # Define colors for each class
        class_colors = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'aqua',
            4: 'magenta',
            5: 'yellow',
            6: 'black'
        }

        # Create a scatter plot with specified colors for each class
        plt.figure(figsize=(10, 8))
        color_index = ['Red', 'Green', 'Blue', 'Aqua', 'Magenta', 'Yellow', 'White']

        for class_label, color in class_colors.items():
            indices = np.where(labels == class_label)
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                        label=f'{color_index[int(class_label)]} digits', color=color)

    else:
        # Define colors for each class
        class_colors = {
            0: 'red',
            1: 'green',
            2: 'blue',
            3: 'aqua',
            4: 'magenta',
            5: 'yellow',
            6: 'black',
            7: 'brown',
            8: 'purple',
            9: 'orange'
        }

        # Create a scatter plot with specified colors for each class
        plt.figure(figsize=(10, 8))

        for class_label, color in class_colors.items():
            indices = np.where(labels == class_label)
            plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1],
                        label=f'{int(class_label)} digits', color=color)

    plt.title(title)
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend()
    plt.show(block=blocking)


data = generate_colored_MNIST()
_, x_test, _, _, _, _ = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet
_, _, _, y_test, _, c_test = (data['Obf'])  # We need the labels without 'to_categorical()' in preprocess
bottleneck = 1024
ObfNet_type = 'mlp'

combined_model = load_model(f'ObfNet_{ObfNet_type}/combined_model_{bottleneck}.h5', compile=False)
combined_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
ObfModel, _ = split_models(combined_model)
print(f"Loaded ObfNet model with Bottleneck: {bottleneck}")

RecNet = load_model(f'RecNet_{ObfNet_type}/RecNet_{bottleneck}.h5', compile=False)
RecNet.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
print(f"Loaded RecNet model cnn with ObfNet BottleNeck {bottleneck}")

x_test_obf = ObfModel.predict(x_test)
x_test_rec = RecNet.predict(x_test_obf)
#tSNE_analysis(x_test, c_test, title=f't-SNE plot of colored MNIST images (raw data)', blocking=False)
tSNE_analysis(x_test_obf, c_test, title=f't-SNE plot of colored MNIST images obfuscated using a {ObfNet_type}-based ObfNet with bottleneck of {bottleneck}', blocking=True)
#tSNE_analysis(x_test_rec, c_test, title=f't-SNE plot of colored MNIST images reconstructed from obfuscated images using a {ObfNet_type}-based ObfNet with bottleneck of {bottleneck}')
