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

sys.path.append('../../motion')

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
                localsRot.append((x,y,z))
                
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

fileToDecode = 'data_rotation_cmu_Euler_2samples_j30_ws120_standardized_scaled10.npz'

X = np.load(fileToDecode)['clips']
mean = np.load(fileToDecode)['mean']
std = np.load(fileToDecode)['std']
scale = np.load(fileToDecode)['scale']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
print(qdata.shape)

X = None

dataSplitPoint = int(len(qdata)*0.8)

#trainingData = array(qdata[0:dataSplitPoint])
#validationData = array(qdata[dataSplitPoint:len(qdata)])
trainingData = qdata

network = load_model('data_rotation_cmu_Euler_2samples_j30_ws120_standardized_scaled10_k15_hu256_vtq2_e1200_d0.15_bz16_valtest0.2_model.h5')

network.compile(optimizer='adam', loss='mse')
network.summary()

print(trainingData.shape)

network.load_weights('data_rotation_cmu_Euler_2samples_j30_ws120_standardized_scaled10_k15_hu256_vtq2_e1200_d0.15_bz16_valtest0.2_weigths.h5')

print('decoding...')

decoded_quat = array(network.predict(trainingData))

print(">MSE I/O NN Q <> QHat:")
print(mse(trainingData, decoded_quat))

mypath = 'data/decoding/'
file = open(mypath+'output.txt', 'w')

onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
folder = onlyfolders[0]
onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
filename = onlyfiles[0]
rootPos, rootRot, localsRot = extractBVHGlobals(mypath+folder+'/'+filename)

anim, names, frametime = BVH.load(mypath+folder+'/'+filename, order='zyx', world=False)

""" Convert to 60 fps (if using 60fps version) """
anim = anim[::2]
BVH.save("original.bvh", anim)


globalRot = anim.rotations[:,0:1]
rotations = anim.rotations[:,1:len(anim.rotations)] #1:len(anim.rotations) to avoid glogal rotation
#rangedRotations = np.array([
#     1,
#     2,  3,  4,  5,
#     7,  8,  9, 10,
#    12, 13, 15, 16,
#    18, 19, 20, 22,
#    25, 26, 27, 29])
#rotations = anim.rotations[:,1:]
#globalRot = anim.rotations[:,0:1] 
print(len(rotations))
reformatRotations = []
print(anim.rotations.shape)
reformatRotationsEuler = []

for frame in rotations:
    joints = []
    jointsEuler = []
    
    for joint in frame:
        joints.append(joint)
        jointsEuler.append(Quaternions(joint).euler().ravel())
        
    reformatRotations.append(joints)
    reformatRotationsEuler.append(jointsEuler)

reformatRotationsEuler = np.array(reformatRotationsEuler)
rotationsA = np.array(reformatRotations)

print(anim.rotations.shape)


rotationsA = rotationsA.reshape(rotationsA.shape[0], rotationsA.shape[1]*rotationsA.shape[2])[0:trainingData[0].shape[0]]
reformatRotationsEuler = reformatRotationsEuler.reshape(reformatRotationsEuler.shape[0], reformatRotationsEuler.shape[1]*reformatRotationsEuler.shape[2])[0:trainingData[0].shape[0]]
print(rotations.shape)
print(">A-R:")
print(np.square(mse(trainingData[0], reformatRotationsEuler)))

#print(">B-R:")
#print(np.square(mse(decoded_quat[0], reformatRotations)))
print(decoded_quat[0].shape)
print(rotationsA.shape)
flatDecoded = decoded_quat.flatten()

decodedlist = []

#originalQIn = (((trainingData[0]*std)+mean)/10)
decodedEulers = ((decoded_quat[0]*std)+mean)

for frame in decodedEulers:
    frameList = []
    for joint in chunks(frame, 3):
        #decodedj = normalize(joint)
        frameList.extend(joint)
        #print(decodedj)
        #print(np.linalg.norm(decodedj))
        #input("press key ")
    decodedlist.append(frameList)

decodedlist = array(decodedlist)

print(">Decoded - RotationsEuler:")
print(np.square(mse(decodedlist, reformatRotationsEuler)))

np.savetxt('QIn.txt', trainingData[0], delimiter=' ') 
#np.savetxt('ScaledIn.txt', trainingData[0], delimiter=' ') 

#decoding
print(decoded_quat.shape)

fullParsedQuat = []
fullParsedEuler = []

idx = 0

decodedlist = array(decodedlist)

outputList = []

for frame in decodedlist:
    if idx != 0:
        file.write('\n')
    
    first = True
    j = 1
    
    frameLine = []
    
    for joint in chunks(frame, 3):
        if first:
            file.write('{0} {1} {2} '.format(rootPos[idx][0], rootPos[idx][1], rootPos[idx][2]))
            file.write('{0} {1} {2} '.format(rootRot[idx][0], rootRot[idx][1], rootRot[idx][2]))
            #frameLine.append(rootRot[idx])
            first = False
            
        
        quateu = np.degrees(np.array(joint))
        #quateu = np.degrees(Quaternions.from_euler(np.array(joint)).euler().ravel())
        frameLine.append(quateu)
        
        file.write('{0} {1} {2} '.format(quateu[0], quateu[1], quateu[2])) #zyx
        
        #if j in rangedRotations:
        #if j != 0:
        anim.rotations[idx][j] = Quaternions.from_euler(np.array(joint), order='zyx')
        
        outputList.append(frameLine)
        j+=1
    idx+=1

#print(outputList[0])
outputList = array(outputList)
print("first shape")
print(outputList.shape)
outputList = outputList.reshape(outputList.shape[0], outputList.shape[1]*outputList.shape[2])
print("second shape")
print(outputList.shape)

print("output list shape:")
print(outputList.shape)
#print(outputList[0].shape)


BVH.save("output.bvh", anim)

fileName = file.name
file.close()

#print(outputList[0:reformatRotationsEuler.shape[0]])
#print(reformatRotationsEuler)

arrayOut = []

with open(fileName) as file:
    arrayOut = [[float(digit) for digit in line.split()[5:-1]] for line in file] #3:-1 remove position

print(">Manual-R:")
print(np.square(mse(arrayOut[:][:], reformatRotationsEuler)))


#qManualOut = np.fromfile(fileName, dtype="float")

#print(anim.rotations.euler().shape)

print("finished")
