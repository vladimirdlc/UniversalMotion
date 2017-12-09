from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, Activation
from keras.models import Model, Sequential, model_from_json
from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np

from keras.layers.normalization import BatchNormalization

from Quaternions import Quaternions

def custom_activation(x):
    return max(K.sigmoid(x) * 5) - 1


import math
from math import radians, degrees
import sys

from itertools import islice

version = "tq2"
fileChanged = "cmu_rotations_full_cmu_30_nonchanged"

X = np.load(fileChanged+".npz")['clips']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = array(X)
X = None

np.random.seed(0)
# split into 80% for train and 20% for test
trainingData, validationData = train_test_split(qdata, test_size=0.20) #shuffle=False


network = Sequential()
degreesOFreedom = trainingData.shape[2] #joints * degreees of freedom
windowSize = trainingData.shape[1] #temporal window 240 frames

kernel_size = 20 #15m

network.add(BatchNormalization(input_shape=(windowSize, degreesOFreedom)))

network.add(Dropout())
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same', strides=2))
network.add(BatchNormalization())

network.add(Dropout(0.2))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same', strides=2))
network.add(BatchNormalization())

hiddenUnits = 512
network.add(Dense(hiddenUnits))
network.add(Dropout(0.2))
network.add(Activation('relu'))
network.add(BatchNormalization())

network.add(UpSampling1D(size=2))

network.add(Dropout(0.2))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same'))
network.add(BatchNormalization())

network.add(UpSampling1D(size=2))

network.add(Dropout(0.2))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='linear', padding='same'))

network.summary()

network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

#network.load_weights('{}_k{}_hu{}_weights.h5'.format(fileChanged,kernel_size,hiddenUnits))

print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=2,
                epochs=50,
                batch_size=200,
                shuffle=True,
                validation_data=(validationData, validationData))
                
network.save_weights('{}_k{}_hu{}_v{}_weights.h5'.format(fileChanged,kernel_size,hiddenUnits, version))
network.save('{}_k{}_hu{}_v{}_model.h5'.format(fileChanged,kernel_size,hiddenUnits, version))

decoded_quat = array(network.predict(trainingData))

print("MSE I/O NN:")
print(np.square(np.subtract(trainingData, decoded_quat)).mean())

print(fileChanged)