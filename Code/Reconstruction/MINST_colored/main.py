import numpy as np

from colored_MNIST_generator import generate_data
from model_trainer import train_inf, preprocess_data

data = generate_data()
x_train, x_test, y_train, y_test = preprocess_data(data['Obf'])  # Use only ObfNet data for both InfNet and ObfNet

# Train InfNet model
InfModel = train_inf(x_train, x_test, y_train, y_test)

