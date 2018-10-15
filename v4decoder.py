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
import eulerangles as eang
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
from enum import Enum

class Decoder(Enum):
    QUATERNION = 'Quaternion'
    ROTATION_MATRIX = 'RotatationMatrix'
    EULER = 'Euler'
    AXIS_ANGLE = 'AxisAngle'

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
            if decodeType is Decoder.QUATERNION:
                joints.append(joint * scale)
            elif decodeType is Decoder.EULER:
                joints.append(Quaternions(joint).euler().ravel() * scale)
            elif decodeType is Decoder.AXIS_ANGLE:
                angle, axis = Quaternions(joint).angle_axis()
                input = axis.flatten()
                input = np.insert(input, 0, angle)
                input = np.array(input)  # 4 values
                joints.append(input * scale)
            elif decodeType is Decoder.ROTATION_MATRIX:
                euler = Quaternions(joint).euler().ravel()  # we get x,y,z
                # eang library uses convention z,y,x
                m = eang.euler2mat(euler[2], euler[1], euler[0])
                input = np.array(m[0].tolist() + m[1].tolist() + m[2].tolist())  # 9 values
                joints.append(input)

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

    return windows

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

#filename = '144_21_parsed'
filename = 'original_144_21_45d'
fullPathAnim = 'data/decoding/' + filename + '.bvh'

print('processing...')

np.set_printoptions(suppress=True)


decodeType = Decoder.QUATERNION #decoding type
fileToDecode = 'cmu_Quat_21_standardized_w480_ws240_normalfps_scaled1000'
X = np.load(fileToDecode+'.npz')
mean = X['mean']
std = X['std']
scale = X['scale']

print('\n')

X = np.array(np.load(fileToDecode+'.npz')['clips'])


np.random.seed(0)
# split into 80% for train and 20% for tests

network = load_model(
    'models/'+fileToDecode+'_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_model.h5')
network.compile(optimizer='adam', loss='mse')
network.summary()

network.load_weights(
    'weights/'+fileToDecode+'_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_weights.h5')

print('decoding...')

#print(">MSE I/O NN Q <> QHat:")
#print(mse(trainingData, decoded_quat))

anim, names, frametime = BVH.load(fullPathAnim, order='zyx', world=False)

BVH.save('output_'+filename+'.bvh', anim)

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
        if decodeType is Decoder.QUATERNION:
            joints.append(joint * scale)
        elif decodeType is Decoder.EULER:
            joints.append(Quaternions(joint).euler().ravel()*scale)
        elif decodeType is Decoder.AXIS_ANGLE:
            angle, axis = Quaternions(joint).angle_axis()
            input = axis.flatten()
            input = np.insert(input, 0, angle)
            input = np.array(input) #4 values
            joints.append(input*scale)
        elif decodeType is Decoder.ROTATION_MATRIX:
            euler = Quaternions(joint).euler().ravel() #we get x,y,z
            #eang library uses convention z,y,x
            m = eang.euler2mat(euler[2], euler[1], euler[0])
            input = np.array(m[0].tolist()+m[1].tolist()+m[2].tolist()) #9 values
            joints.append(input)

    reformatRotations.append(joints)

reformatRotations = np.array(reformatRotations)

datatypeLength = X.shape[3] #4 for quaternions

X = process_file_rotations(fullPathAnim, window=X.shape[1], window_step=X.shape[1])
X = np.array(list(X))
print(X.shape)
X -= mean
X /= std
#X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])

decoded_quat = array(network.predict(X))

rotations = (((decoded_quat)*std)+mean)/scale

idx = 0

#go by all windows 240
for wdw in rotations:
    for frame in wdw:
        if idx >= anim.rotations.shape[0]:
            break

        j = 0
        frameLine = []
        for joint in chunks(frame, datatypeLength):
            if decodeType is Decoder.QUATERNION:
                anim.rotations[idx][j] = Quaternions(joint)
            elif decodeType is Decoder.EULER:
                joint = [joint[2], joint[1], joint[0]]
                anim.rotations[idx][j] = Quaternions.from_euler(np.array(joint), order='zyx')
            elif decodeType is Decoder.AXIS_ANGLE:
                z, y, x = eang.angle_axis2euler(joint[0], [joint[1], joint[2], joint[3]]) #theta, x, y, z
                joint = np.degrees([z, y, x])  # in z,y,x format
                joints.append(joint)
            elif decodeType is Decoder.ROTATION_MATRIX:
                m0 = np.array([joint[0], joint[1], joint[2]])
                m1 = np.array([joint[3], joint[4], joint[5]])
                m2 = np.array([joint[6], joint[7], joint[8]])
                m = [m0, m1, m2]
                joint = eang.mat2euler(m)  # in z,y,x rad format
                joints.append(joint)
            j += 1
        idx += 1

BVH.save(filename+'_'+decodeType.value+'.bvh', anim)

print("finished")
