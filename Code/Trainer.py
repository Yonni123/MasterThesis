import Train
import TrainingSettings as TS
import Utils
import Helper
import Logger


ts = TS.TrainingSettings()  # default settings
ts.train_dir = "data/imagenet/train"
ts.val_dir = "data/imagenet/val"

# load model
ResNet = Utils.get_ResNet50()
ObfNet = Utils.get_obfmodel((224, 224, 3), 100)
model = Utils.join_models(ObfNet, ResNet)

# train model
data = Train.loadData(ts)
record = Train.train(model, data, ts)

if ts.save:
    Helper.savetofile(ts.save_dir+"data.hist", record.history)
    model.save(ts.save_dir+"final.h5"),