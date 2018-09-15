from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json, load_model
from sklearn import preprocessing

from numpy import array
import numpy as np
import csv

from Quaternions import Quaternions

import math
from math import radians, degrees
import sys

from itertools import islice
import eulerangles as eang

sys.path.insert(0, './motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots


def extractBVHGlobals(fullPath):
    print("Processing... "+fullPath)

    file = open(fullPath, 'r')
    print("opened")

    data = []
    frameDataStart = False

    for line in file:
        line = line.strip()
        if not line: continue

        if line.startswith('Frame '):
            frameDataStart = True
            continue
        if frameDataStart:
            data.append(line)

    rootPos = []
    rootRot = []
    localsRot = []

    for currentFrame in data:
        quaternionData = []
        floats = [float(x) for x in currentFrame.split()]
        first = True
        second = True
        
        for x,y,z in zip(*[iter(floats)]*3):
            if first:
                rootPos.append((x,y,z))
                first = False
            else:
                if second:
                    rootRot.append((x,y,z))
                    second = False
                localsRot.append((z,y,x))
                
    file.close()
    return rootPos, rootRot, localsRot
    
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

print('processing...')

fileToDecode = 'cmu_rotations_full_rotmat_30_standardized_w240_ws120_normalfps_scaled10000000.npz'

X = np.load(fileToDecode)['clips']
mean = np.load(fileToDecode)['mean']
std = np.load(fileToDecode)['std']
scale = np.load(fileToDecode)['scale']

#print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
#print(qdata.shape)

dataSplitPoint = int(len(qdata)*0.8)

trainingData = qdata

network = load_model('models/cmu_rotations_full_rotmat_30_standardized_w240_ws120_normalfps_scaled10000000_k15_hu256_vtq2_e600_d0.15_bz1_valtest0.2_model.h5')

network.compile(optimizer='adam', loss='mse')
network.summary()

#print(trainingData.shape)

network.load_weights('weights/cmu_rotations_full_rotmat_30_standardized_w240_ws120_normalfps_scaled10000000_k15_hu256_vtq2_e600_d0.15_bz1_valtest0.2_weigths.h5')

print('decoding...')

decodedMatrix = network.predict(trainingData)
#print(decodedMatrix.shape)
print(">MSE I/O NN Q <> QHat:")
print(mse(trainingData, decodedMatrix))
decoded = ((decodedMatrix[0]*std)+mean) #first only

#denormalizing matrix data from [0, 1] to [-1,1]

mypath = 'data/decoding/'
file = open(mypath+'output.txt', 'w')

onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
folder = onlyfolders[0]
onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
filename = onlyfiles[0]
rootPos, rootRot, localsRot = extractBVHGlobals(mypath+folder+'/'+filename)

anim, names, frametime = BVH.load(mypath+folder+'/'+filename, order='zyx', world=True)

""" Convert to 60 fps (if using 60fps version) """
#anim = anim[::2]

#rotations = anim.rotations[:,1:len(anim.rotations)] #1:len(anim.rotations) to avoid glogal rotation
#print(rotations.shape)
reformatRotationsMatrix = []

reformatEulerDecodedRotMat = []

for frame in decoded:
    joints = []
    jointsMatrix = []
    for mat in zip(*[iter(frame)]*9):
        print(mat)
        mat /= scale
        m0 = np.array([mat[0], mat[1], mat[2]])
        m1 = np.array([mat[3], mat[4], mat[5]])
        m2 = np.array([mat[6], mat[7], mat[8]])
        m = [m0, m1, m2]
        print(m)
        print('real e:')
        print('from m:')
        joint = eang.mat2euler(m) #in z,y,x rad format
        jointsMatrix.append(joint)
        
    reformatEulerDecodedRotMat.append(jointsMatrix)

reformatEulerDecodedRotMat = np.array(reformatEulerDecodedRotMat)

#decoding
idx = 0

outputList = []

for frame in reformatEulerDecodedRotMat:
    first = True
    second = True
    j = 1
    
    frameLine = []
    
    for joint in frame:
        #if first:
        #    first = False
        #    continue

        print(joint)
        anim.rotations[idx][j] = Quaternions.from_euler(np.array(joint), order='zyx')
        j+=1
    idx+=1

BVH.save("output.bvh", anim)

fileName = file.name
file.close()