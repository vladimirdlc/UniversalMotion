from Quaternion import Quat
from Quaternion import normalize
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json
#from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np


from Quaternions import Quaternions

import math
from math import radians, degrees
import sys

from itertools import islice

X = np.load('data_rotation_cmu_quat.npz')['clips']
print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
#X = np.swapaxes(X, 1, 2).astype('float32')

#X = (X+1)*0.5
qdata = array(X)
X = None

dataSplitPoint = int(len(qdata)*0.8)

trainingData = array(qdata[0:dataSplitPoint])
validationData = array(qdata[dataSplitPoint:len(qdata)])

network = Sequential()
degreesOFreedom = trainingData.shape[1] #joints * degreees of freedom
windowSize = trainingData.shape[2] #temporal window 240 frames

input_frame = Input(shape=(degreesOFreedom, windowSize))  # adapt this if using `channels_first` image data format
filterSize = 30

#---
network.add(Dense(256, input_shape=(degreesOFreedom, windowSize)))
#network.add(Dropout(0.25, input_shape=(degreesOFreedom, windowSize)))
#network.add(Dense(128, input_shape=(degreesOFreedom, windowSize)))
#network.add(Dense(input_shape=(degreesOFreedom, windowSize)))
kernel_size = 25

network.add(Conv1D(256, kernel_size, use_bias=True, activation='relu', padding='same'))

network.add(MaxPooling1D(padding='same'))

#network.add(Dropout(0.25))

network.add(Conv1D(windowSize, kernel_size, use_bias=True, activation='relu', padding='same'))

network.add(UpSampling1D(size=2))

network.save("test_model.h5")
print("saving model")
#network.add(Activation('relu'))
'''

autoencoder = Model(input_frame, decoded)
autoencoder.summary()

# load weights into new model
autoencoder.load_weights("autoencoder_weights.h5")
autoencoder.save("test_model.h5")
#print("Loaded model from disk")
'''
network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=2,
                epochs=50,
                batch_size=128,
                shuffle=True,
                validation_data=(validationData, validationData))
                
network.save_weights('autoencoder_weights.h5')

print("finished")