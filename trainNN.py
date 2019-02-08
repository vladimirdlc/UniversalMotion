import logging
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, Activation
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, Activation
from keras.models import Model, Sequential, model_from_json
from keras import optimizers
from sklearn.model_selection import train_test_split
from numpy import array
from plotCallback import PlotLoss

import numpy as np

import time
import random

then = time.time() #Time before the operations start


import math
from math import radians, degrees
import sys
import keras as K
from itertools import islice

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.2f}'.format})
np.random.seed(0)

version = "tq3"
fileChanged = "cmu_Euler_21j_w240x120"

print('started processing {}', fileChanged)
X = np.load(fileChanged+".npz")['clips']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = array(X)
X = None

# split into 80% for train and 20% for test
trainingData, validationData = train_test_split(qdata, test_size=0.2)

network = Sequential()
degreesOFreedom = trainingData.shape[2] #joints * degreees of freedom
windowSize = trainingData.shape[1] #temporal window 240 frames

kernel_size = 25
dropoutAmount = 0.15
hiddenUnits = 256

activationType = 'relu'

network.add(Dropout(dropoutAmount, input_shape=(windowSize, degreesOFreedom)))

network.add(Conv1D(hiddenUnits, kernel_size, activation=activationType, use_bias=True, padding='same'))

network.add(Dropout(dropoutAmount, input_shape=(windowSize, hiddenUnits)))
network.add(Conv1D(degreesOFreedom, kernel_size, activation='linear', use_bias=True, padding='same'))

network.summary()

epochs = 600

myadam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

network.compile(optimizer=myadam, loss='mse')

batch_size = 128

idPrefix = '{}_k{}_hu{}_e{}_d{}_bz{}_valtest0.2_activation{}'.format(fileChanged, kernel_size, hiddenUnits, epochs, dropoutAmount, batch_size, activationType)

plot_losses = PlotLoss(epochs, 'results/'+idPrefix)

print(trainingData.shape)

history_callback = network.fit(trainingData, trainingData, verbose=2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[plot_losses],
                validation_data=(validationData, validationData))
                
print('hu{}'.format(hiddenUnits))
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt('results/{}_lossHistory.txt'.format(idPrefix), numpy_loss_history, fmt='%f')

val_loss_history = np.array(val_loss_history)
np.savetxt('results/{}_valLossHistory.txt'.format(idPrefix), val_loss_history, fmt='%f')

network.save_weights('weights/{}_weights.h5'.format(idPrefix))
network.save('models/{}_model.h5'.format(idPrefix))

decoded_quat = array(network.predict(trainingData))

print("MSE I/O NN:")
print(np.square(np.subtract(trainingData, decoded_quat)).mean())

print("finished")
print(fileChanged)

now = time.time() #Time after it finished
print("It took: ", now-then, " seconds")