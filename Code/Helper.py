from sys import platform
import pickle
import datetime

DATASET_MAIN_PATH = 'dataset/'

def formatURL(url):
    if platform == 'linux' or platform == 'linux2':
        return url.replace('\\','/', 10)
    return url

def savetofile(filename, data):
    with open(filename, 'wb') as pickle_file:
        pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)

def loadFromFile(filename):
    with open(filename, 'rb') as pickle_load:
        return pickle.load(pickle_load)

def timepostfix():
    time = datetime.datetime.now()
    return f'{time.year:02d}{time.month:02d}{time.day:02d}_{time.hour:02d}{time.minute:02d}'

def getTime():
    return ''+str(datetime.datetime.now())