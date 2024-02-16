from matplotlib import pyplot as plt
from data_generator import generate_colored_MNIST
from model_trainer import *
import os

data = generate_colored_MNIST()
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

print('\n***ColorNet***')
x_train, x_test, y_train, y_test, c_train, c_test = preprocess_data(data['Rec'])  # Switch to using the other half of the data
color_models_cnn = []
cnn_accuracies = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train CNN based ColorNets and store them in the color_models_mlp array
    obf_model = obf_models_cnn[counter]
    x_test_obf = obf_model.predict(x_test, verbose=0)
    if os.path.exists(f'ColorNet_cnn/ColorNet_{bn}.h5'):
        ColorNet = load_model(f'ColorNet_cnn/ColorNet_{bn}.h5', compile=False)
        ColorNet.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        loss, accuracy = ColorNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
        print(f"Loaded ColorNet for Obf_cnn model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Obf_Bottleneck: {bn}")
    else:
        print(f'Training ColorNet for Obf_cnn model with Obf_Bottleneck {bn}...')
        ColorNet = train_color_model_mlp(
            ground_truth_train_images=x_train,  # These images were never used during the creation of Inf and Obf nets
            ground_truth_test_images=x_test,
            c_train=c_train,    # The color information
            c_test=c_test,
            obf_model=obf_model,    # The CNN based obfnet
            save_path = f'ColorNet_cnn/ColorNet_{bn}.h5'
        )
        loss, accuracy = ColorNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
    color_models_cnn.append(ColorNet)
    cnn_accuracies.append(accuracy * 100)
    counter += 1

color_models_mlp = []
mlp_accuracies = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train MLP based ColorNets and store them in the color_models_mlp array
    obf_model = obf_models_mlp[counter]
    x_test_obf = obf_model.predict(x_test, verbose=0)
    if os.path.exists(f'ColorNet_mlp/ColorNet_{bn}.h5'):
        ColorNet = load_model(f'ColorNet_mlp/ColorNet_{bn}.h5', compile=False)
        ColorNet.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        loss, accuracy = ColorNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
        print(f"Loaded ColorNet for Obf_mlp model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Obf_Bottleneck: {bn}")
    else:
        print(f'Training ColorNet for Obf_mlp model with Obf_Bottleneck {bn}...')
        ColorNet = train_color_model_mlp(
            ground_truth_train_images=x_train,  # These images were never used during the creation of Inf and Obf nets
            ground_truth_test_images=x_test,
            c_train=c_train,    # The color information
            c_test=c_test,
            obf_model=obf_model,    # The MLP based obfnet
            save_path = f'ColorNet_mlp/ColorNet_{bn}.h5'
        )
        loss, accuracy = ColorNet.evaluate(x=x_test_obf, y=c_test, verbose=0)
    mlp_accuracies.append(accuracy * 100)
    color_models_mlp.append(ColorNet)
    counter += 1

# Plotting the accuracies over bottlenecks
plt.plot(bottlenecks, mlp_accuracies, label='MLP-Based ObfNet')
plt.plot(bottlenecks, cnn_accuracies, label='CNN-Based ObfNet')

# Adding labels and title
plt.xlabel('Bottleneck')
plt.ylabel('Accuracy %')
plt.title('Accuracy over Bottleneck using ColorNet-MLP')

# Adding legend
plt.legend()

# Display the plot
plt.show()