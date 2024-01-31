import Train
import TrainingSettings as TS
from Helper import Utils
import Helper
import PATHS
import ObfNet_Desgins as OND
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import preprocess_input


ts = TS.TrainingSettings()  # Default training settings
ts.preprocess_function = preprocess_input
ts.save = False
ts.train_dir = PATHS.IMAGENET10c_PATH + 'train'
ts.val_dir = PATHS.IMAGENET10c_PATH + 'val'
ts.epochs = 100
ts.batch_size = 64
ts.learning_rate = 1e-2

# load model
ResNet = Utils.get_pretrained_ResNet50(PATHS.RESNET_WEIGHTS)
ObfNet = OND.deconv((224, 224, 3))
model = Utils.join_models(ObfNet, ResNet)
Utils.freeze_inf_model(model)    # Freeze the inference model

print("Model Summary")
model.summary()

history = Train.train(model, ts)