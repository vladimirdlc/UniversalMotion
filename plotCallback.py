import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation

import numpy as np
from matplotlib import pyplot as plt
from IPython.display import clear_output

class PlotLoss(keras.callbacks.Callback):
    ymax = 0.025
    ymin = 0.0

    def __init__(self, epochs=-1, path=""):
        self.epochs = epochs
        self.path = path+'_losstPlot.png'
        plt.ion()
    
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.ylim(ymax=self.ymax)
        plt.ylim(ymin=self.ymin)
        plt.legend()
        plt.show()
        plt.pause(0.0001)

        if self.epochs-1 == epoch:
            plt.savefig(self.path)
        
        plt.gcf().clear()
    
