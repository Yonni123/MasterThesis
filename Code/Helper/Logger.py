import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.applications.resnet50 as resnet50
import os

import Helper

class DummyLogger:
    def log(self, string):
        pass
    def logValue(self, name, value):
        pass
    def addCat(self, string):
        pass
    def addCat2(self, string):
        pass
    def logSummary(self, model):
        pass

class Logger:
    def __init__(self, path, fname):
        self.path = path
        self.fname = fname
        self.fullPath = Helper.formatURL(path+"/"+fname+".log")
    def log(self, string):
        with open(self.fullPath, 'a') as f:
            f.write(f"{string}\n")

    def logValue(self, name, value):
        self.log(name+" = "+str(value))

    def addCat(self,string):
        with open(self.fullPath, 'a') as f:
            f.write("\n\n\n")
            f.write("="*80+"\n")
            f.write("    "+string+"\n")
            f.write("="*80+"\n")

    def addCat2(self,string):
        with open(self.fullPath, 'a') as f:
            f.write("\n")
            f.write("  "+string+":\n")

    def logSummary(self, model):
        self.addCat("Model Summary")
        with open(self.fullPath, 'a') as f:
            model.summary(print_fn=lambda s: f.write(s+'\n'))