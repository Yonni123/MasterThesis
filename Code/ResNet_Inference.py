from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from matplotlib import pyplot as plt
from tensorflow.keras.utils import load_img
import numpy as np
import os

#img_path = 'Random_Images/fish.jpg'
img_path = 'D:\ImageNet\Imagenet_Small\\ahh'

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load an image (replace 'path_to_your_image.jpg' with your image file)
# If path is not a directory, assume it is an image file
if not os.path.isdir(img_path):
    img = np.array(load_img(img_path, target_size=(224, 224)))
    img = np.expand_dims(img, axis=0)
else:
    # If path is a directory, assume it is a directory of images and load them all in shape (-1, 224, 224, 3)
    img = []
    for filename in os.listdir(img_path):
        image = np.array(load_img(img_path + '/' + filename, target_size=(224, 224)))
        img.append(image)
    img = np.array(img)

# Make predictions
img = preprocess_input(img)
predictions = model.predict(img)

# Decode and print the top 3 predicted classes
decoded_predictions = decode_predictions(predictions, top=3)

for i, prediction in enumerate(decoded_predictions):
    print(f'Prediction {i + 1}:')
    for (pred_class, pred_description, pred_confidence) in prediction:
        print(f'- {pred_class}: {pred_description} ({pred_confidence * 100:.2f}%)')
