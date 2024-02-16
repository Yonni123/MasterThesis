import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from data_generator import generate_colored_MNIST, generate_solid_data
from model_trainer import *
from RecNet_visualizer import visualize_recnets
import os

data = generate_colored_MNIST()
x_train, x_test, y_train, y_test, _, _ = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet

print('\n***InfNet***')
# Load the InfModel or train it from scratch, this is a CNN-based InfNet
if os.path.exists('inf_model.h5'):
    InfModel = load_model('inf_model.h5', compile=False)
    InfModel.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    loss, accuracy = InfModel.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Loaded InfNet model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
else:
    print('Training CNN-based inference model...')
    InfModel = train_inf_model(x_train, x_test, y_train, y_test)

print('\n***ObfNet***')
bottlenecks = [8, 16, 32, 64, 128, 256, 512, 1024]    # Different ObfNet bottlenecks to measure feature leakage
obf_models_cnn = []
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    if os.path.exists(f'ObfNet_cnn/combined_model_{bn}.h5'):
        combined_model = load_model(f'ObfNet_cnn/combined_model_{bn}.h5', compile=False)
        combined_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        loss, accuracy = combined_model.evaluate(x=x_test, y=y_test, verbose=0)
        ObfModel, _ = split_models(combined_model)
        print(f"Loaded ObfNet model cnn with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Bottleneck: {bn}")
    else:
        print('Training ObfNet CNN-based model...')
        ObfModel = train_obf_model_cnn(x_train, x_test, y_train, y_test, InfModel, bottleneck=bn)
    obf_models_cnn.append(ObfModel)

obf_models_mlp = []
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    if os.path.exists(f'ObfNet_mlp/combined_model_{bn}.h5'):
        combined_model = load_model(f'ObfNet_mlp/combined_model_{bn}.h5', compile=False)
        combined_model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        loss, accuracy = combined_model.evaluate(x=x_test, y=y_test, verbose=0)
        ObfModel, _ = split_models(combined_model)
        print(f"Loaded ObfNet model mlp with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Bottleneck: {bn}")
    else:
        print('Training ObfNet MLP-based model...')
        ObfModel = train_obf_model_mlp(x_train, x_test, y_train, y_test, InfModel, bottleneck=bn)
    obf_models_mlp.append(ObfModel)

print('\n***SolidNet***')
######################################
######################################
# Keep in mind that ALL RecNets are of type MLP with bottleneck 1024
# They have the same architecture as ObfNet MLP with bottleneck 1024
# They are called different names like MLP and CNN because each RecNet
# correspond to a different ObfNet, please don't be confused by it
######################################
######################################
x_train, x_test, y_train, y_test, c_train, c_test = preprocess_data(data['Rec'])  # Switch to using the other half of the data
random_color_images = generate_solid_data(N=30000)  # Generate 30k images with random solid colors of shape (30k, 28, 28, 3)
print('SolidNet - CNN')
solid_models_cnn = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train SolidNet
    if os.path.exists(f'SolidNet_cnn/SolidNet_{bn}.h5'):
        SolidNet = load_model(f'SolidNet_cnn/SolidNet_{bn}.h5', compile=False)
        SolidNet.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print(f"Loaded SolidNet model cnn with ObfNet BottleNeck {bn}")
    else:
        print(f'Training SolidNet CNN-based model with ObfNet BottleNeck {bn}...')
        SolidNet = train_rec_model_mlp(
            ground_truth_train_images=random_color_images,  # Try reversing ObfNet with just random color images
            ground_truth_test_images=x_test,
            obf_model=obf_models_cnn[counter],
            save_path = f'SolidNet_cnn/SolidNet_{bn}.h5'
        )
    solid_models_cnn.append(SolidNet)
    counter += 1

print('SolidNet - MLP')
solid_models_mlp = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train SolidNet
    if os.path.exists(f'SolidNet_mlp/SolidNet_{bn}.h5'):
        SolidNet = load_model(f'SolidNet_mlp/SolidNet_{bn}.h5', compile=False)
        SolidNet.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        print(f"Loaded SolidNet model mlp with ObfNet BottleNeck {bn}")
    else:
        print(f'Training SolidNet MLP-based model with ObfNet BottleNeck {bn}...')
        SolidNet = train_rec_model_mlp(
            ground_truth_train_images=random_color_images,  # Try reversing ObfNet with just random color images
            ground_truth_test_images=x_test,
            obf_model=obf_models_mlp[counter],
            save_path = f'SolidNet_mlp/SolidNet_{bn}.h5'
        )
    solid_models_mlp.append(SolidNet)
    counter += 1

visualize_recnets(obf_models_mlp, solid_models_mlp, data, names=[
    ['ObfNet_MLP_8', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_16', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_32', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_64', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_128', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_256', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_512', 'SolidNet_MLP_1024'],
    ['ObfNet_MLP_1024', 'SolidNet_MLP_1024'],
])
visualize_recnets(obf_models_mlp, solid_models_cnn, data, names=[
    ['ObfNet_CNN_8', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_16', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_32', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_64', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_128', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_256', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_512', 'SolidNet_MLP_1024'],
    ['ObfNet_CNN_1024', 'SolidNet_MLP_1024'],
])
