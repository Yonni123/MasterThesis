import Train
import TrainingSettings as TS
from Helper import Utils
import Helper
import PATHS
import ObfNet_Desgins as OND
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input


ts = TS.TrainingSettings()  # Default training settings
ts.train_dir = PATHS.IMAGENET10c_PATH + 'train'
ts.val_dir = PATHS.IMAGENET10c_PATH + 'val'
ts.epochs = 12
ts.batch_size = 64

# load model
model = Utils.get_pretrained_ResNet50()
# ObfNet = OND.deconv((224, 224, 3))
# model = Utils.join_models(ObfNet, ResNet)
# Utils.freeze_inf_model(model)    # Freeze the inference model

print("Model Summary")
model.summary()

history = Train.train(model, ts)