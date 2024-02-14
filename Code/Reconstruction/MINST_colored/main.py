import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from colored_MNIST_generator import generate_data
from model_trainer import *
from RecNet_visualizer import visualize_recnets

data = generate_data()
x_train, x_test, y_train, y_test, _, _ = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet

print('\n***InfNet***')
# Load the InfModel or train it from scratch, this is a CNN-based InfNet
try:
    InfModel = load_model('inf_model.h5')
    loss, accuracy = InfModel.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Loaded InfNet model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
except:
    print('Training CNN-based inference model...')
    InfModel = train_inf_model(x_train, x_test, y_train, y_test)

print('\n***ObfNet***')
bottlenecks = [8, 16, 32, 64, 128, 256, 512, 1024]    # Different ObfNet bottlenecks to measure feature leakage
obf_models_cnn = []
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    try:
        combined_model = load_model(f'ObfNet_cnn/combined_model_{bn}.h5')
        loss, accuracy = combined_model.evaluate(x=x_test, y=y_test, verbose=0)
        ObfModel, _ = split_models(combined_model)
        print(f"Loaded ObfNet model cnn with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Bottleneck: {bn}")
    except:
        print('Training ObfNet CNN-based model...')
        ObfModel = train_obf_model_cnn(x_train, x_test, y_train, y_test, InfModel, bottleneck=bn)
    obf_models_cnn.append(ObfModel)

obf_models_mlp = []
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    try:
        combined_model = load_model(f'ObfNet_mlp/combined_model_{bn}.h5')
        loss, accuracy = combined_model.evaluate(x=x_test, y=y_test, verbose=0)
        ObfModel, _ = split_models(combined_model)
        print(f"Loaded ObfNet model mlp with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Bottleneck: {bn}")
    except:
        print('Training ObfNet MLP-based model...')
        ObfModel = train_obf_model_mlp(x_train, x_test, y_train, y_test, InfModel, bottleneck=bn)
    obf_models_mlp.append(ObfModel)

print('\n***RecNet***')
######################################
######################################
# Keep in mind that ALL RecNets are of type MLP with bottleneck 1024
# They have the same architecture as ObfNet MLP with bottleneck 1024
# They are called different names like MLP and CNN because each RecNet
# correspond to a different ObfNet, please don't be confused by it
######################################
######################################
x_train, x_test, y_train, y_test, _, _ = preprocess_data(data['Rec'])  # Switch to using the other half of the data
print('RecNet - CNN')
rec_models_cnn = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    try:
        RecNet = load_model(f'RecNet_cnn/RecNet_{bn}.h5')
        print(f"Loaded RecNet model cnn with ObfNet BottleNeck {bn}")
    except:
        print(f'Training RecNet CNN-based model with ObfNet BottleNeck {bn}...')
        RecNet = train_rec_model_mlp(
            ground_truth_train_images=x_train,  # These images were never used during the creation of Inf and Obf nets
            ground_truth_test_images=x_test,
            obf_model=obf_models_cnn[counter],
            save_path = f'RecNet_cnn/RecNet_{bn}.h5'
        )
    rec_models_cnn.append(RecNet)
    counter += 1

print('RecNet - MLP')
rec_models_mlp = []
counter = 0 # This is just for indexing the ObfNets
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    try:
        RecNet = load_model(f'RecNet_mlp/RecNet_{bn}.h5')
        print(f"Loaded RecNet model mlp with ObfNet BottleNeck {bn}")
    except:
        print(f'Training RecNet MLP-based model with ObfNet BottleNeck {bn}...')
        RecNet = train_rec_model_mlp(
            ground_truth_train_images=x_train,  # These images were never used during the creation of Inf and Obf nets
            ground_truth_test_images=x_test,
            obf_model=obf_models_mlp[counter],
            save_path = f'RecNet_mlp/RecNet_{bn}.h5'
        )
    rec_models_mlp.append(RecNet)
    counter += 1

visualize_recnets(obf_models_mlp, rec_models_mlp, data, names=[
    ['ObfNet_MLP_8', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_16', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_32', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_64', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_128', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_256', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_512', 'RecNet_MLP_1024'],
    ['ObfNet_MLP_1024', 'RecNet_MLP_1024'],
])
visualize_recnets(obf_models_mlp, rec_models_mlp, data, names=[
    ['ObfNet_CNN_8', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_16', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_32', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_64', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_128', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_256', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_512', 'RecNet_MLP_1024'],
    ['ObfNet_CNN_1024', 'RecNet_MLP_1024'],
])
