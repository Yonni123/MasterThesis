import numpy as np
from tensorflow.keras.models import load_model
from colored_MNIST_generator import generate_data
from model_trainer import train_inf_model, preprocess_data, train_obf_model_cnn, join_models, split_models

data = generate_data()
x_train, x_test, y_train, y_test = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet

# Load the InfModel or train it from scratch, this is a CNN-based InfNet
try:
    InfModel = load_model('inf_model.h5')
    loss, accuracy = InfModel.evaluate(x=x_test, y=y_test, verbose=0)
    print(f"Loaded InfNet model with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}")
except:
    print('Training CNN-based inference model...')
    InfModel = train_inf_model(x_train, x_test, y_train, y_test)


bottlenecks = [8, 16, 32, 64, 128, 256, 512]    # Different ObfNet bottlenecks to measure feature leakage
obf_models_cnn = []
for bn in bottlenecks:  # Train CNN based ObfNets and store them in the obf_models_cnn array
    try:
        combined_model = load_model(f'ObfNet_cnn/combined_model_{bn}.h5')
        loss, accuracy = InfModel.evaluate(x=x_test, y=y_test, verbose=0)
        ObfModel, _ = split_models(combined_model)
        print(f"Loaded ObfNet model cnn with Accuracy: {accuracy * 100:.2f}%, Loss: {loss:.4f}, Bottleneck: {bn}")
    except:
        print('Training ObfNet-based inference model...')
        ObfModel = train_obf_model_cnn(x_train, x_test, y_train, y_test, InfModel, bottleneck=bn)
    obf_models_cnn.append(ObfModel)



