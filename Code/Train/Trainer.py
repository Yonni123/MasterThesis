import Train
import TrainingSettings as TS
from Code.Helper import Utils
from Code import PATHS
from Code.Designs.cnn import lightweight_cnn_1, lightweight_cnn_2, \
    arch_1, currently_testing, up_sample, create_autoencoder_1
from Code.Designs.ObfNet_Desgins import *
from Code.Designs.Diffused_ObfNet import diffused_ObfNet
from tensorflow.keras.applications.resnet import preprocess_input
from tensorflow.keras.models import Sequential, load_model, Model, clone_model


ts = TS.TrainingSettings()  # Default training settings
ts.preprocess_function = preprocess_input
ts.train_dir = PATHS.IMAGENET10c_PATH + 'train'
ts.val_dir = PATHS.IMAGENET10c_PATH + 'val'
ts.epochs = 100
ts.batch_size = 16
ts.learning_rate = 1e-3

# load model
ResNet = Utils.get_pretrained_ResNet50(PATHS.RESNET_WEIGHTS)
#ObfNet = create_autoencoder_1((224, 224, 3))
ObfNet = load_model('C:\MT\Code\History\session_20240204_191415\\best_obf_weights.h5')
ObfNet.summary()
model = Utils.join_models(ObfNet, ResNet)
Utils.freeze_inf_model(model)    # Freeze the inference model

print("Model Summary")
model.summary()
Train.train(model, ts)