import os

import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from colored_MNIST_generator import generate_data, generate_noisy_data
from model_trainer import *

data = generate_data()
x_train, x_test, y_train, y_test, c_train, c_test = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet

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

print('\n***NoisyNet***')
x_train, x_test, y_train, y_test, c_train, c_test = preprocess_data(data['Rec'])  # Switch to using the other half of the data
noisy_data_generated = False

noisy_models_cnn = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train CNN based ColorNets and store them in the color_models_mlp array
    obf_model = obf_models_cnn[counter]
    if os.path.exists(f'NoisyNet_cnn/NoisyNet_{bn}.h5'):
        NoisyNet = load_model(f'NoisyNet_cnn/NoisyNet_{bn}.h5')
        x_test_obf = obf_model.predict(x_test, verbose=0)
        loss, accuracy = NoisyNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
        print(f"Loaded NoisyNet for Obf_cnn model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Obf_Bottleneck: {bn}")
    else:
        print(f'Training NoisyNet for Obf_cnn model with Obf_Bottleneck {bn}...')
        if not noisy_data_generated:
            noisy_images, colors = generate_noisy_data(N=30000)  # Generate 30k noisy data to train the NoisyNet on
            colors = to_categorical(colors)
            noisy_data_generated = True
        NoisyNet = train_color_model_mlp(
            ground_truth_train_images=noisy_images,
            ground_truth_test_images=x_test,
            c_train=colors,    # The color information
            c_test=c_test,
            obf_model=obf_model,    # The CNN based obfnet
            save_path = f'NoisyNet_cnn/NoisyNet_{bn}.h5'
        )
    noisy_models_cnn.append(NoisyNet)
    counter += 1

noisy_models_mlp = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train MLP based ColorNets and store them in the color_models_mlp array
    obf_model = obf_models_mlp[counter]
    if os.path.exists(f'NoisyNet_mlp/NoisyNet_{bn}.h5'):
        NoisyNet = load_model(f'NoisyNet_mlp/NoisyNet_{bn}.h5')
        x_test_obf = obf_model.predict(x_test, verbose=0)
        loss, accuracy = NoisyNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
        print(f"Loaded NoisyNet for Obf_mlp model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Obf_Bottleneck: {bn}")
    else:
        print(f'Training NoisyNet for Obf_mlp model with Obf_Bottleneck {bn}...')
        if not noisy_data_generated:
            noisy_images, colors = generate_noisy_data(N=30000)  # Generate 30k noisy data to train the NoisyNet on
            colors = to_categorical(colors)
            noisy_data_generated = True
        NoisyNet = train_color_model_mlp(
            ground_truth_train_images=noisy_images,
            ground_truth_test_images=x_test,
            c_train=colors,    # The color information
            c_test=c_test,
            obf_model=obf_model,    # The MLP based obfnet
            save_path = f'NoisyNet_mlp/NoisyNet_{bn}.h5'
        )
    noisy_models_mlp.append(NoisyNet)
    counter += 1