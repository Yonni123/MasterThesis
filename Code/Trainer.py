import Train
import TrainingSettings as TS
import Utils
import Helper
import Logger


ts = TS.TrainingSettings()  # default settings
ts.train_dir = "data/imagenet/train"
ts.val_dir = "data/imagenet/val"

# Design ObfNet
ts.obf_layers = [
                 ('Conv2D', [3, 3]), ('MaxPooling2D', [2, 2]), # Reduce image size to 112x112
                 ('Conv2D', [3, 3]), ('MaxPooling2D', [2, 2]), # Reduce image size to 56x56
                 ('Conv2D', [3, 3]), ('MaxPooling2D', [2, 2]), # Reduce image size to 28x28
                 ('Dense', [1000])
]

# load model
ResNet = Utils.get_pretrained_ResNet50()
ObfNet = Utils.get_obfmodel(ts)
model = Utils.join_models(ObfNet, ResNet)
Utils.freeze_infmodel(model)    # Freeze the inference model

print("Model Summary")
model.summary()
ResNet.summary()
# train model
data = Train.loadData(ts)
record = Train.train(model, data, ts)

if ts.save:
    Helper.savetofile(ts.save_dir+"data.hist", record.history)
    model.save(ts.save_dir+"final.h5"),