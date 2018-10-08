from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D, BatchNormalization, Activation
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
fileToDecode = 'cmu_rotations_Quat_cmu_20_standardized_w240_ws120_normalfps_scaled1000.npz' #'cmu_rotations_full_cmu_30_w240_2samples_standardized_scaled10000.npz'

np.set_printoptions(suppress=True)

X = np.load(fileToDecode)['clips']
mean = np.load(fileToDecode)['mean']
std = np.load(fileToDecode)['std']
scale = np.load(fileToDecode)['scale']

print(X.shape)

X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])

qdata = np.array(X[0:200]) #extracting only the BVH section we want to test
print(qdata.shape)

X = None

dataSplitPoint = int(len(qdata)*0.2)

#trainingData = array(qdata[dataSplitPoint:-1])
#validationData = array(qdata[dataSplitPoint:len(qdata)])
np.random.seed(0)
# split into 80% for train and 20% for tests
trainingData = qdata

network = load_model('models/cmu_rotations_Quat_cmu_20_standardized_w240_ws120_normalfps_scaled1000_k25_hu256_vtq2_e600_d0.15_bz128_classic_valtest0.2_activationrelu_model.h5')
network.compile(optimizer='adam', loss='mse')
network.summary()

print(trainingData.shape)

network.load_weights('weights/cmu_rotations_Quat_cmu_20_standardized_w240_ws120_normalfps_scaled1000_k25_hu256_vtq2_e600_d0.15_bz128_classic_valtest0.2_activationrelu_weights.h5')

print('decoding...')

decoded_quat = array(network.predict(trainingData))

print(">MSE I/O NN Q <> QHat:")
print(mse(trainingData, decoded_quat))

mypath = 'data/decoding/'

onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]
folder = onlyfolders[0]
onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
filename = onlyfiles[0]

anim, names, frametime = BVH.load(mypath+folder+'/'+filename, order='zyx', world=False)

BVH.save('original.bvh', anim)

""" Convert to 60 fps """

#globalRot = anim.rotations[:,0:1]
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

print(anim.rotations.shape)

decoded_quat = ((decoded_quat*std)+mean)

flatDecoded = decoded_quat.flatten()

decodedlistNorm = []

for frame in decoded_quat[0]:
    frameList = []
    for joint in chunks(frame, 4):

        decodedj = normalize(joint*scale)
        frameList.extend(decodedj)
        #print(decodedj)
        #print(np.linalg.norm(decodedj))
        #input("press key ")
    decodedlistNorm.append(frameList)

decodedlistNorm = array(decodedlistNorm)

print(">Diff dec[0] and Norm(B)")
print(np.square(mse(decodedlistNorm, decoded_quat[0])))

#print(">Norma(B)-R:")
#print(np.square(mse(decodedlistNorm, rotationsA)))

#np.savetxt('QIn.txt', trainingData[0], delimiter=' ') 
#np.savetxt('QHat.txt', decoded_quat[0], delimiter=' ')
#np.savetxt('QBar.txt', decodedlistNorm, delimiter=' ')  
#np.savetxt('ScaledIn.txt', trainingData[0], delimiter=' ') 

#decoding
print(decoded_quat.shape)

fullParsedQuat = []  
fullParsedEuler = []

idx = 0

decodedlistNorm = array(decodedlistNorm)

for frame in decodedlistNorm:    
    first = True
    j = 1
    
    frameLine = []
    
    for joint in chunks(frame, 4):

        #if j in rangedRotations:
        anim.rotations[idx][j] = Quaternions(joint)
        j+=1
    idx+=1

BVH.save("output.bvh", anim)


#print(outputList[0:reformatRotationsEuler.shape[0]])
#print(reformatRotationsEuler)

#print(">Manual-R:")
#print(np.square(mse(arrayOut[:][:], reformatRotationsEuler)))


#qManualOut = np.fromfile(fileName, dtype="float")

#print(anim.rotations.euler().shape)

print("finished")
