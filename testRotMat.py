import logging
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, Activation
from keras.models import Model, Sequential, model_from_json
from sklearn.model_selection import train_test_split
from numpy import array
from plotCallback import PlotLoss

import numpy as np

import math
from math import radians, degrees
import sys
import keras as K
from itertools import islice

np.set_printoptions(suppress=True,
   formatter={'float_kind':'{:0.2f}'.format})
np.random.seed(0)

version = "tq2"
fileChanged = "cmu_rotations_full_rotmat_30_standardized_w240_ws120_normalfps_scaled1000"

print('started processing {}', fileChanged)
X = np.load(fileChanged+".npz")['clips']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = array(X)
X = None

# split into 80% for train and 20% for test
#trainingData = qdata
trainingData, validationData = train_test_split(qdata[1:-1], test_size=0.2)
dataSplitPoint = int(len(qdata)*0.2)

#validationData = array(qdata[0:dataSplitPoint])
#trainingData = array(qdata[0,dataSplitPoint:-1])
validationData = validationData.tolist()
validationData.append(qdata[0])
validationData = np.array(validationData)
#trainingData = qdata
#testData = testData.reshape([], testData.shape[0], testData.shape[1])

#print(validationData.shape)
#print(testFile.shape)
network = Sequential()
degreesOFreedom = trainingData.shape[2] #joints * degreees of freedom
windowSize = trainingData.shape[1] #temporal window 240 frames

kernel_size = 15
dropoutAmount = 0.15
hiddenUnits = 256

network.add(Dropout(dropoutAmount, input_shape=(windowSize, degreesOFreedom)))

network.add(Conv1D(hiddenUnits, kernel_size, activation='sigmoid', use_bias=True, padding='same'))

network.add(Dropout(dropoutAmount, input_shape=(windowSize, hiddenUnits)))
network.add(Conv1D(degreesOFreedom, kernel_size, activation='linear', use_bias=True, padding='same'))

network.summary()


epochs = 600

network.compile(optimizer='adam', loss='mse')

batch_size = 16
#network.load_weights('cmu_rotations_full_cmu_30_w240_standardized_scaled10000_k15_hu512_vtq2_e400_d0.25_bz1_weigths.h5')

idPrefix = '{}_k{}_hu{}_v{}_e{}_d{}_bz{}_valtest0.2'.format(fileChanged,kernel_size,hiddenUnits, version, epochs, dropoutAmount, batch_size)

plot_losses = PlotLoss(epochs, 'results/'+idPrefix)

print(trainingData.shape)
#print(validationData.shape)
history_callback = network.fit(trainingData, trainingData, verbose=2,
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[plot_losses],
                validation_data=(validationData, validationData))
                
print('hu{}'.format(hiddenUnits))
loss_history = history_callback.history["loss"]
val_loss_history = history_callback.history["val_loss"]

numpy_loss_history = np.array(loss_history)
np.savetxt('results/{}_lossHistory.txt'.format(idPrefix), numpy_loss_history, delimiter=', ')

numpy_loss_history = np.array(val_loss_history)
np.savetxt('results/{}_valLossHistory.txt'.format(idPrefix), numpy_loss_history, delimiter=', ')


network.save_weights('{}_weigths.h5'.format(idPrefix))
network.save('{}_model.h5'.format(idPrefix))

decoded_quat = array(network.predict(trainingData))

print("MSE I/O NN:")
print(np.square(np.subtract(trainingData, decoded_quat)).mean())

print("finished")
print(fileChanged)