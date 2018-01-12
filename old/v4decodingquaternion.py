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
npzFile = 'cmu_rotations_full_cmu_30_w240_standardized_scaled10000.npz'
X = np.load(npzFile)['clips']
mean = np.load(npzFile)['mean']
std = np.load(npzFile)['std']

print("Mean")
print(mean)
print("std")
print(std)

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
print(qdata.shape)

X = None

#dataSplitPoint = int(len(qdata)*0.8)
#trainingData = array(qdata[0:dataSplitPoint])
#validationData = array(qdata[dataSplitPoint:len(qdata)])
trainingData = qdata

network = load_model('cmu_rotations_full_cmu_30_w240_standardized_scaled10000_k15_hu512_vtq2_e300_d0.25_weigths.h5_model.h5')

network.compile(optimizer='adam', loss='mse')
network.summary()

print(trainingData.shape)

network.load_weights('cmu_rotations_full_cmu_30_w240_standardized_scaled10000_k15_hu512_vtq2_e300_d0.25_weigths.h5_weigths.h5')

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

""" Convert to 60 fps """
#anim = anim[::2]
BVH.save("original.bvh", anim)


globalRot = anim.rotations[:,0:1]
rotations = anim.rotations[:,0:len(anim.rotations)] #1:len(anim.rotations) to avoid glogal rotation
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
print(">ScaledQIn-R:")

originalQIn = (((trainingData[0]*std)+mean))#/10)
decodedUpscaled = (((decoded_quat[0]*std)+mean))#/10)
print(np.square(mse(originalQIn, rotationsA)))

print(">ScaledHat-R:")
print(np.square(mse(originalQIn, rotationsA)))
print(decodedUpscaled.shape)
print(rotationsA.shape)
flatDecoded = decoded_quat.flatten()

decodedlistNorm = []
print(decodedUpscaled)
print("----")
#for frame in decoded_quat[0]:
for frame in decodedUpscaled:
    frameList = []
    for quat in chunks(frame, 4):
        #print(quat)
        #print(quat.shape)
        #print('*')
        #print(quat)
        decodedj = normalize(quat)
        frameList.extend(decodedj)
        #print(decodedj)
        #print(np.linalg.norm(decodedj))
        #input("press key ")
    decodedlistNorm.append(frameList)

decodedlistNorm = array(decodedlistNorm)

print(">Diff dec[0] and Norm(B)")
print(np.square(mse(decodedlistNorm, decoded_quat[0])))

print(">Norma(B)-R:")
print(np.square(mse(decodedlistNorm, rotationsA)))

np.savetxt('QIn.csv', originalQIn, delimiter=',') 
np.savetxt('QHat.csv', decoded_quat[0], delimiter=',')
np.savetxt('SHatDecodedUpscaled.csv', decodedUpscaled, delimiter=',')
np.savetxt('QBar.csv', decodedlistNorm, delimiter=',')  
#np.savetxt('ScaledIn.txt', trainingData[0], delimiter=' ') 

#decoding
print(decoded_quat.shape)

fullParsedQuat = []
fullParsedEuler = []

idx = 0

decodedlistNorm = array(decodedlistNorm)

outputList = []

for frame in decodedlistNorm:
    if idx != 0:
        file.write('\n')
    
    first = True
    j = 0
    
    frameLine = []
    
    for joint in chunks(frame, 4):
        if first:
            file.write('{0} {1} {2} '.format(rootPos[idx][0], rootPos[idx][1], rootPos[idx][2]))
            file.write('{0} {1} {2} '.format(rootRot[idx][0], rootRot[idx][1], rootRot[idx][2]))
            #frameLine.append(rootRot[idx])
            first = False
            
        quateu = np.degrees(Quaternions(joint).euler().ravel())
        frameLine.append(quateu)
        
        file.write('{0} {1} {2} '.format(quateu[2], quateu[1], quateu[0]))
        
        #if j in rangedRotations:
        if j != 0:
            anim.rotations[idx][j] = Quaternions(joint)
        
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

#with open(fileName) as file:
#    arrayOut = [[float(digit) for digit in line.split()[5:-1]] for line in file] #3:-1 remove position

print(">Manual-R:")
print(np.square(mse(arrayOut[:][:], reformatRotationsEuler)))


#qManualOut = np.fromfile(fileName, dtype="float")

#print(anim.rotations.euler().shape)

print("finished")
