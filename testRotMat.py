from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json
from numpy import array
import numpy as np


from Quaternions import Quaternions

import math
from math import radians, degrees
import sys

from itertools import islice

X = np.load('data_rotation_6rotmat.npz')['clips']
print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = array(X)
X = None

dataSplitPoint = int(len(qdata)*0.8)

trainingData = array(qdata[0:dataSplitPoint])
validationData = array(qdata[dataSplitPoint:len(qdata)])

network = Sequential()
degreesOFreedom = trainingData.shape[1] #joints * degreees of freedom
windowSize = trainingData.shape[2] #temporal window 240 frames

input_frame = Input(shape=(degreesOFreedom, windowSize))  # adapt this if using `channels_first` image data format

network.add(Dense(256, input_shape=(degreesOFreedom, windowSize)))

kernel_size = 5

network.add(Conv1D(256, kernel_size, use_bias=True, activation='relu', padding='same'))

network.add(MaxPooling1D(padding='same'))

network.add(Conv1D(windowSize, kernel_size, use_bias=True, activation='relu', padding='same'))

network.add(UpSampling1D(size=2))

network.load_weights('autoencoder_weights_rot.h5')
network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

network.save("test_model_rot.h5")
print("saving model")

network.summary()

print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=2,
                epochs=200,
                batch_size=128,
                shuffle=True,
                validation_data=(validationData, validationData))
                
network.save_weights('autoencoder_weights_rot.h5')

print("finished")