from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import load_img
import numpy as np

# Load the pre-trained ResNet50 model
model = ResNet50(weights='imagenet')

# Load an image (replace 'path_to_your_image.jpg' with your image file)
img_path = 'Random_Images/fish.jpg'
img = load_img(img_path, target_size=(224, 224))

# Preprocess the image to fit ResNet50 input requirements
x = np.array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Make predictions
predictions = model.predict(x)

# Decode and print the top 3 predicted classes
decoded_predictions = decode_predictions(predictions, top=3)[0]
for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
    print(f"{i + 1}: {label} ({score:.2f})")
