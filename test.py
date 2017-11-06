from Quaternion import Quat
from Quaternion import normalize
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential, model_from_json
#from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np

from Quaternions import Quaternions



import math
from math import radians, degrees
import sys

from itertools import islice

X = np.load('data_rotation_full_cmu.npz')['clips']

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
network.add(Dropout(0.25, input_shape=(degreesOFreedom, windowSize)))

kernel_size = 25
network.add(Conv1D(256, kernel_size, use_bias=True, activation='tanh', padding='same'))
network.add(MaxPooling1D(padding='same'))

network.add(Dropout(0.25))

network.add(Conv1D(windowSize, kernel_size, use_bias=True, activation='tanh', padding='same'))

network.add(UpSampling1D(size=2))

network.summary()


# load weights into new model
#network.load_weights("autoencoder_weights.h5")
#print("Loaded model from disk")

network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=2,
                epochs=200,
                batch_size=1000,
                shuffle=True,
                validation_data=(validationData, validationData))
                
network.save_weights('autoencoder_weights.h5')

print("finished")