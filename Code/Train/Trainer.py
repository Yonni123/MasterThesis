import Train
import TrainingSettings as TS
from Helper import Utils
import Helper
import PATHS
import ObfNet_Desgins as OND
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input


train_dir = PATHS.IMAGENET_PATH + 'train'
val_dir = PATHS.IMAGENET_PATH + 'val'

# load model
ResNet = Utils.get_pretrained_ResNet50()
ObfNet = OND.deconv((224, 224, 3))
model = Utils.join_models(ObfNet, ResNet)
Utils.freeze_inf_model(model)    # Freeze the inference model

print("Model Summary")
model.summary()

# Set the image size based on your model's input size
img_size = (224, 224)

# Set batch size and number of epochs
batch_size = 32
epochs = 10

# Create data generators with data augmentation for training set
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

# Data generator for validation set (only rescaling)
val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Flow training images in batches using the generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'  # or 'binary' depending on your problem
)

# Flow validation images in batches using the generators
val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',  # or 'binary_crossentropy'
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=val_generator,
    validation_steps=val_generator.samples // batch_size
)

# Save the trained model
model.save('your_model.h5')
