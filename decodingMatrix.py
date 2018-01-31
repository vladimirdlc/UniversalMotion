from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json, load_model

from numpy import array
import numpy as np

from Quaternions import Quaternions

import math
from math import radians, degrees
import sys

from itertools import islice

import eulerangles as eang

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

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
                else: continue
                
    file.close()
    return rootPos, rootRot


X = np.load('data_rotation_6rotmat.npz')['clips']
print(X.shape)

sizeDecoded = X.shape[3]

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
print(qdata.shape)

X = None

dataSplitPoint = int(len(qdata)*0.8)

trainingData = array(qdata[0:dataSplitPoint])
validationData = array(qdata[dataSplitPoint:len(qdata)])

network = load_model('test_model_rot.h5')

print(trainingData.shape)
print(validationData.shape)

network.load_weights('autoencoder_weights_rot.h5')

network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
network.summary()


decoded = array(network.predict(trainingData))

#denormalizing matrix data from [0, 1] to [-1,1]
decoded = (decoded*2)-1

mypath = 'data/decoding/'
file = open(mypath+'output.txt', 'w')

onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
folder = onlyfolders[0]
onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
filename = onlyfiles[0]
rootPos, rootRot = extractBVHGlobals(mypath+folder+'/'+filename)

for windowData in decoded:
    for frame in windowData:
        file.write('\n')
        first = True
        idx = 0

        previousJoint = []
        for a,b,c,d,e,f in zip(*[iter(frame)]*6):
            m0 = np.array([a, b, c])
            m1 = np.array([d, e, f])
            m2 = np.cross(m0, m1)
            m1 = np.cross(m2, m0)
            
            m = [m0, m1, m2]
            
            
            if first:
                file.write('{0} {1} {2} '.format(rootPos[idx][0],rootPos[idx][1],rootPos[idx][2]))
                file.write('{0} {1} {2} '.format(rootRot[idx][0],rootRot[idx][1],rootRot[idx][2]))
                first = False
                idx += 1
            
            joint = np.degrees(eang.mat2euler(m)) #in z,y,x format

            #instead of xyz we're doing zyx cause of the CMU
            file.write('{0} {1} {2} '.format(joint[0], joint[1], joint[2]))
        break


file.close()


print("finished")
