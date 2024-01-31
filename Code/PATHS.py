import sys
import os

sys.path.insert(1, 'Helper')
sys.path.insert(2, 'Privacy_Measures')
sys.path.insert(3, 'Train')

PROJECT_MAIN_DIR = os.path.dirname(os.path.abspath(__file__)) + '\\'
RESNET_WEIGHTS = PROJECT_MAIN_DIR + 'History\ResNet10c\session_20240130_133837_final.h5'
IMAGENET_PATH = 'C:\ImageNet\\'
IMAGENET10c_PATH = 'C:\ImageNet10c\\'
HISTORY_DIR = PROJECT_MAIN_DIR + 'History/'  # path to save model and logs and weights
