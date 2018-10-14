import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, \
    BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json, load_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from numpy import array
import numpy as np
import csv

from Quaternions import Quaternions

import math
from math import radians, degrees
import sys

from itertools import islice

sys.path.insert(0, './motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
import Animation as Animation

def mse(a, b):
    return np.square(np.subtract(a, b)).mean()


def normalize(array):
    """
    Normalize a 4 element array/list/numpy.array for use as a quaternion

    :param quat_array: 4 element list/array
    :returns: normalized array
    :rtype: numpy array
    """
    quat = np.array(array)
    return quat / np.sqrt(np.dot(quat, quat))


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def process_file_rotations(filename, window=240, window_step=240):
    anim, names, frametime = BVH.load(filename, order='zyx')

    """ Convert to 60 fps """
    # anim = anim[::2]
    """ Do FK """
    print(len(anim.rotations))

    """ Remove Uneeded Joints """
    # exported
    rotations = anim.rotations[:, 0:len(anim.rotations)]

    print(len(rotations))
    """ Remove Uneeded Joints """
    reformatRotations = []

    # encoding
    for frame in rotations:
        joints = []

        for joint in frame:
            # print(joint)
            joints.append(joint * scale)

        reformatRotations.append(joints)


    rotations = np.array(reformatRotations)
    rotations = rotations.reshape(rotations.shape[0], rotations.shape[1]*rotations.shape[2])

    #    (2448, 21, 4) -> (2448, 84)
    print(rotations.shape)

    """ Slide over windows """
    windows = []
    partition = []

    for rot in rotations:
        partition.append(rot)
        if len(partition) >= window:
            windows.append(partition)
            partition = []


    #print(np.array(windows).shape)
    #print(windows.shape)

    #for j in range(0, len(rotations)) :


    '''
    for j in range(0, len(rotations) - window // 8, window_step):
        # input(j)
        """ If slice too small pad out by repeating start and end poses """
        slice = rotations[j:j + window]

        if len(slice) < window:
            left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)

        if len(slice) != window: raise Exception()

        windows.append(slice)
    '''
    return windows

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

filename = 'original_144_21_45d'
fullPathAnim = 'data/decoding/' + filename + '.bvh'

print('processing...')
fileToDecode = 'cmu_Quat_21_standardized_w480_ws240_normalfps_scaled1000.npz'  # 'cmu_rotations_full_cmu_30_w240_2samples_standardized_scaled10000.npz'

np.set_printoptions(suppress=True)

X = np.load(fileToDecode)
mean = X['mean']
std = X['std']
scale = X['scale']
#startidx = X['filesinfo'].item()[filename]['startidx']
#endidx = X['filesinfo'].item()[filename]['endidx']

print(X['filesinfo'])

print('\n')
# print(a.shape)
# startidx = X['filesinfo'][filename, 'startidx']
# endidx = X['filesinfo'][filename, 'endidx']

X = np.array(np.load(fileToDecode)['clips'])

#X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])

#qdata = np.array(X[startidx:endidx])  # extracting only the BVH section we want to test
#print(qdata.shape)

np.random.seed(0)
# split into 80% for train and 20% for tests

network = load_model(
    'models/cmu_Quat_21_standardized_w480_ws240_normalfps_scaled1000_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_model.h5')
network.compile(optimizer='adam', loss='mse')
network.summary()

network.load_weights(
    'weights/cmu_Quat_21_standardized_w480_ws240_normalfps_scaled1000_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_weights.h5')

print('decoding...')

#print(">MSE I/O NN Q <> QHat:")
#print(mse(trainingData, decoded_quat))

anim, names, frametime = BVH.load(fullPathAnim, order='zyx', world=False)

BVH.save('original.bvh', anim)

""" Convert to 60 fps """

# globalRot = anim.rotations[:,0:1]
rotations = anim.rotations[:, 0:len(anim.rotations)]  # 1:len(anim.rotations) to avoid glogal rotation

print(len(rotations))
""" Remove Uneeded Joints """
reformatRotations = []

# encoding
for frame in rotations:
    joints = []

    for joint in frame:
        # print(joint)
        joints.append(joint * scale)

    reformatRotations.append(joints)

reformatRotations = np.array(reformatRotations)

X = process_file_rotations(fullPathAnim, window=X.shape[1], window_step=X.shape[1])
X = np.array(list(X))
print(X.shape)
X -= mean
X /= std
#X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])

decoded_quat = array(network.predict(X))

rotations = (((decoded_quat)*std)+mean)/scale
print(rotations[0])
print(anim.rotations.shape)

useHipsIdentity = False #useful for denoising or using original

idx = 0

#go by all windows 240
for wdw in rotations:
    for frame in wdw:
        if idx >= anim.rotations.shape[0]:
            break
        j = 0

        frameLine = []
        for joint in chunks(frame, 4):
            anim.rotations[idx][j] = Quaternions(joint)
            j += 1
        idx += 1

BVH.save("output.bvh", anim)

# print(outputList[0:reformatRotationsEuler.shape[0]])
# print(reformatRotationsEuler)

# print(">Manual-R:")
# print(np.square(mse(arrayOut[:][:], reformatRotationsEuler)))


# qManualOut = np.fromfile(fileName, dtype="float")

# print(anim.rotations.euler().shape)

print("finished")
