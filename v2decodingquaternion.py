from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
from keras.models import Model, Sequential, model_from_json, load_model
from sklearn import preprocessing

from numpy import array
import numpy as np

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
    
#denormalizing quaternions data from [0, 1] to [-1,1]
def scaleUpFromNN(joint):
    return (joint*2)-1

def mse(a, b):
    return np.square(np.subtract(a, b)).mean()
    
#def normalize(joint):
    #print(joint / np.sqrt(np.dot(joint, joint)))
    #input("Press Enter to continue...")
def normalize(array):
   """ 
   Normalize a 4 element array/list/numpy.array for use as a quaternion
   
   :param quat_array: 4 element list/array
   :returns: normalized array
   :rtype: numpy array
   """
   quat = np.array(array)
   return quat / np.sqrt(np.dot(quat, quat))
    #return v
    #return joint / np.sqrt(np.dot(joint, joint))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

print('processing...')
X = np.load('data_rotation_cmu_quat_30.npz')['clips']
print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
print(qdata.shape)

X = None

dataSplitPoint = int(len(qdata)*0.8)

#trainingData = array(qdata[0:dataSplitPoint])
#validationData = array(qdata[dataSplitPoint:len(qdata)])
trainingData = qdata

network = load_model('data_rotation_cmu_quat_30_model_v2_128_clean_20k_500.h5')

network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
network.summary()

print(trainingData.shape)
#print(validationData.shape)

network.load_weights('data_rotation_cmu_quat_30_weights_128_clean_20k_500.h5')

print('decoding...')

decoded_quat = array(network.predict(trainingData))

print(">MSE I/O NN:")
print(mse(trainingData, decoded_quat))

print(">MSE I/O NN Decoded:")
print(mse(scaleUpFromNN(trainingData), scaleUpFromNN(decoded_quat)))

#denormalizing quaternions data from [0, 1] to [-1,1]
mypath = 'data/decoding/'
file = open(mypath+'output.txt', 'w')

onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
folder = onlyfolders[0]
onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
filename = onlyfiles[0]
rootPos, rootRot, localsRot = extractBVHGlobals(mypath+folder+'/'+filename)

anim, names, frametime = BVH.load(mypath+folder+'/'+filename, order='zyx', world=False)
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
for frame in rotations:
    joints = []
    for joint in frame:
        joints.append(joint)
    reformatRotations.append(joints)

rotationsA = np.array(reformatRotations)

print(anim.rotations.shape)


rotationsA = rotationsA.reshape(rotationsA.shape[0], rotationsA.shape[1]*rotationsA.shape[2])[0:trainingData[0].shape[0]]
#rotationsA = rotations.reshape(rotations.shape[0], rotations.shape[1]*rotations.shape[2])
print("work")
print(rotations.shape)
print(">A-R:")
print(np.square(mse(scaleUpFromNN(trainingData[0]), rotationsA)))

print(">B-R:")
print(np.square(mse(scaleUpFromNN(decoded_quat[0]), rotationsA)))
print(decoded_quat[0].shape)
print(rotationsA.shape)
flatDecoded = decoded_quat.flatten()

decodedlistNorm = []

for frame in decoded_quat[0]:
    frameList = []
    for joint in chunks(frame, 4):
        decodedj = normalize(scaleUpFromNN(joint))
        print(np.linalg.norm(decodedj))
        frameList.extend(decodedj)
        print(decodedj)
        #input("press key ")
    decodedlistNorm.append(frameList)

decodedlistNorm = array(decodedlistNorm)

print(">Diff dec[0] and Norm(B)")
print(np.square(mse(decodedlistNorm, decoded_quat[0])))

print(">Norma(B)-R:")
print(np.square(mse(decodedlistNorm, rotationsA)))

#globalRot = anim.rotations[:,0:1]
#fullRotations = np.concatenate((globalRot, decodedlistNorm), axis=1)

np.savetxt('ScaledHat.txt', decoded_quat[0], delimiter=' ')
np.savetxt('QHat.txt', scaleUpFromNN(decoded_quat[0]), delimiter=' ')
np.savetxt('QBar.txt', decodedlistNorm, delimiter=' ')  
np.savetxt('QIn.txt', scaleUpFromNN(trainingData[0]), delimiter=' ') 
np.savetxt('ScaledIn.txt', trainingData[0], delimiter=' ') 

i = 0
#decoding
print(decoded_quat.shape)

'''
for frame in decoded_quat[0]:
    j = 0
    print(i)
    for joint in chunks(frame, 4):
        rotations[i][j+1] = Quaternions(np.array(joint))
        j+=1
    i+=1
'''

#print(rotationsB.shape)
print(">A-B:")
#print(np.square(np.subtract(rotationsA, rotationsB)).mean())
#print(">A-R:")
#print(np.square(np.subtract(rotationsA, rotations)).mean())
#fullRotationsB = np.concatenate((globalRot, reformatRotations), axis=1)


xparsedQuat = []
ydecodedQuat = []
xparsedEuler = []
ydecodedEuler = []
fullParsedQuat = []
fullParsedEuler = []


k = 1
rotationsNew = rotations

idx = 0

decodedlistNorm = array(decodedlistNorm)

for frame in decodedlistNorm:
    first = True
    file.write('\n')
    j = 0
    
    for joint in chunks(frame, 4):
        #if j in rangedRotations:
        if j != 0:
            anim.rotations[idx][j] = Quaternions(joint)
        j+=1
        
    idx+=1
'''if first:
            print(rootPos[idx])
            print(rootRot[idx])
            file.write('{0} {1} {2} '.format(rootPos[idx][0], rootPos[idx][1], rootPos[idx][2]))
            file.write('{0} {1} {2} '.format(rootRot[idx][0], rootRot[idx][1], rootRot[idx][2]))
            first = False
        '''

        #anim.rotations[idx][j] = Quaternions(joint)
        #if first:
            #first = False
        #else:
            

        #quateu = np.degrees(Quaternions(joint).euler().ravel())
        #file.write('{0} {1} {2} '.format(quateu[2], quateu[1], quateu[0]))
    #idx += 1

#anim.rotations = rotations
BVH.save("testsmall.bvh", anim)


'''
for windowData in decoded_quat:
    if k >= 240:
        break
        
    for frame in windowData:
        file.write('\n')
        first = True
        idx = 0
        
        if k >= 240:
            break
        
        joints = []
        previousJoint = []
        for x,y,z,w in zip(*[iter(frame)]*4):
            joint = np.array([x, y, z, w])
            if first:
                file.write('{0} {1} {2} '.format(rootPos[idx][0],rootPos[idx][1],rootPos[idx][2]))
                file.write('{0} {1} {2} '.format(rootRot[idx][0],rootRot[idx][1],rootRot[idx][2]))

                fullParsedQuat.append(Quaternions.from_euler(np.array(rootRot[idx]), order='zyx').ravel())
                xyzjoint = Quaternions.from_euler(np.array(rootRot[idx]), order='zyx').euler().ravel()
                
                fullParsedEuler.append((xyzjoint[0], xyzjoint[2], xyzjoint[1]))

                first = False
                idx += 1
                #continue #31 (30 joints + global rot continue)
                
            rotationsNew[k] = Quaternions(joint)
            xyzjoint = Quaternions.from_euler(np.array(localsRot[j]), order='zyx').euler().ravel()
            xparsedQuat.append(Quaternions.from_euler(np.array(localsRot[j]), order='zyx').ravel())
            ydecodedQuat.append(Quaternions(joint).ravel())
            xparsedEuler.append((xyzjoint[0], xyzjoint[2], xyzjoint[1]))
            fullParsedEuler.append((xyzjoint[0], xyzjoint[2], xyzjoint[1]))
            decodedEuler = Quaternions(joint).euler().ravel()
            ydecodedEuler.append((decodedEuler[2], decodedEuler[1], decodedEuler[0]))

            #normalize [-1,1] #test not normalizing
            joint = normalize(joint)
            
            #unraveling depending on quat data
            if len(previousJoint) == 0:
                previousJoint = joint
                joint = scaleUpFromNN(joint)
            else:
                distance1 = np.linalg.norm(joint-previousJoint)
                distance2 = np.linalg.norm(-joint-previousJoint)
                
                if distance1 > distance2:
                    joint = scaleUpFromNN(-joint)
                else:
                    joint = scaleUpFromNN(joint)
            
            #joint = scaleUpFromNN(joint)
            k += 1
            
            quateu = np.degrees(Quaternions(rotations[j][idx]).euler().ravel())

            #instead of xyz we're doing zyx cause of the CMU
            file.write('{0} {1} {2} '.format(quateu[2], quateu[1], quateu[0]))
            
            j += 1

        break
        
'''
file.close()

print(">MSE  Quaternion Parsed/Decoded:")
print(mse(scaleUpFromNN(np.array(xparsedQuat)), scaleUpFromNN(np.array(ydecodedQuat))))

print(">MSE Euler Parsed/Decoded:")
print(mse(xparsedEuler, ydecodedEuler))

fullParsedQuat = np.array(fullParsedQuat)
print(trainingData[0].shape)
print(fullParsedQuat.shape)
tdata = trainingData[0][0:len(fullParsedQuat)]
#tdata = tdata.reshape(fullParsedQuat.shape)

print(">MSE Euler Trained/Decoded:")
#print(mse(tdata, fullParsedQuat))


print("finished")
