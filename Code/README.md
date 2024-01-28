# _ObfNet_(SourceCode)
This folder contains the source code from the original paper of ObfNet, found in [this git repo](https://github.com/ntu-aiot/ObfNet/tree/master).

# Helper
This directory contains helper files that has helper functions in them
- **Evaluator** - used to evaluate models and for now, returns accuracy and loss
- **Image_Modifier** - it can resize and modify images in a directory, used to clean up ImageNet
- **Logger** - still not programmed yet but it's supposed to log everything during training
- **Utils** - responsible for functions to join obfnet and infnet and split them and a bit more

# History
A directory to save training history, weights in h5 files and document model architectures, date and accuracy, and other stuff using the logger

# Images
This folder contains example images of any random stuff from MNIST to a random image of a lion

# Privacy measures
Has all the files needed to measure privacy in images, from structural similarity to HaarPSI  
Might not be the best privacy measures for ObfNet specifically, but that will be decided later

# Train
This directory has all the files needed to train a model