from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, Activation
from keras.models import Model, Sequential, model_from_json
from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras import regularizers

from Quaternions import Quaternions

#test tanh, diff kernel and HU and Removing BN
import math
from math import radians, degrees
import sys
import keras as K
from itertools import islice

version = "tq2"
fileChanged = "cmu_rotations_full_cmu_30_w240_standardized_scaled10000"

X = np.load(fileChanged+".npz")['clips']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = array(X)
X = None

np.random.seed(0)
# split into 80% for train and 20% for test
trainingData, validationData = train_test_split(qdata, test_size=0.2)


network = Sequential()
degreesOFreedom = trainingData.shape[2] #joints * degreees of freedom
windowSize = trainingData.shape[1] #temporal window 240 frames

kernel_size = 15
dropoutAmount = 0.25
hiddenUnits = 512
"""
#network.add(BatchNormalization(input_shape=(windowSize, degreesOFreedom)))

network.add(Dropout(dropoutAmount, input_shape=(windowSize, degreesOFreedom)))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same', strides=2))
#network.add(BatchNormalization())

#network.add(Dropout(dropoutAmount))
#network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same', strides=2))
#network.add(BatchNormalization())

#hiddenUnits = 512
#network.add(Dense(hiddenUnits))
network.add(Dropout(dropoutAmount))
network.add(Activation('relu'))
#network.add(BatchNormalization())

network.add(UpSampling1D(size=2))

network.add(Dropout(dropoutAmount))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='relu', padding='same'))
#network.add(BatchNormalization())

network.add(UpSampling1D(size=2))

network.add(Dropout(dropoutAmount))
network.add(Conv1D(degreesOFreedom, kernel_size, use_bias=True, activation='linear', padding='same'))
"""
network.add(Dropout(dropoutAmount, input_shape=(windowSize, degreesOFreedom)))
network.add(Conv1D(hiddenUnits, kernel_size, activation='sigmoid', use_bias=True, padding='same'))

network.add(Dropout(dropoutAmount, input_shape=(windowSize, hiddenUnits)))
network.add(Conv1D(degreesOFreedom, kernel_size, activation='linear', use_bias=True, padding='same'))

network.summary()


epochs = 1000

'''
learning_rate = 0.01
decay_rate = learning_rate / epochs
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)
'''

network.compile(optimizer='adam', loss='mse')

network.load_weights('cmu_rotations_full_cmu_30_w240_standardized_scaled10000_k15_hu512_vtq2_e100_d0.25_weigths.h5')

print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=2,
                epochs=epochs,
                batch_size=16,
                validation_data=(validationData, validationData))

network.save_weights('{}_k{}_hu{}_v{}_e{}_d{}_weigths.h5'.format(fileChanged,kernel_size,hiddenUnits, version, epochs, dropoutAmount))
network.save('{}_k{}_hu{}_v{}_e{}_d{}_model.h5'.format(fileChanged,kernel_size,hiddenUnits, version, epochs, dropoutAmount))

decoded_quat = array(network.predict(trainingData))

print("MSE I/O NN:")
print(np.square(np.subtract(trainingData, decoded_quat)).mean())

print("finished")
print(fileChanged)