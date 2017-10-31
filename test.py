from Quaternion import Quat
from Quaternion import normalize
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential, model_from_json
#from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np


import math
from math import radians, degrees
import sys

from itertools import islice

def window(seq, n=2):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result

def processBVH(fullPath):
        print("Processing... "+fullPath)
        file = open(fullPath, 'r')

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

        framesQData = []
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
                elif second:
                    rootRot.append((x,y,z))
                    second = False
                else:
                    quat = Quat((x,y,z))

                    if quat.q[3] < 0:
                        quaternionData.append(-quat.q)
                    else:
                        quaternionData.append(quat.q)

            framesQData.append(quaternionData)

        file.close()
        
        return framesQData, rootPos, rootRot

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def writeDecodedToFile(decodedArray, file, file2):
    i = 0
    print("writing to file")
    sizeDecoded = len(decodedArray)
    
    for windowData in decodedArray:
        i += 1
        print("{} of {} ".format(i,sizeDecoded))

        for frame in windowData:
            file.write('\n')
            file2.write('\n')
            
            first = True
            for x,y,z,w in zip(*[iter(frame)]*4):
                if first:
                     file.write('{0} {1} {2} {3} {4} {5} '.format(0, 0, 0, 0, 0, 0))
                     first = False

                quat = Quat((x,y,z, w))
                file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
                
                file2.write('{0} {1} {2} {3} '.format(x, y, z, w))


mypath = 'data/CMU/testing/'
onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

qdata = []
rootPos = []
rootRot = []

targetFrames = 25

for folder in onlyfolders:
    onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
    k = 0
    for filename in onlyfiles:


        qFrame, qRootPos, qRootRot = processBVH(mypath+folder+'/'+filename)
        
        qdata.append(array(qFrame))
        k += 1;
        
        if k == 8:
            qdata = array(qdata)
            dataSplitPoint = int(len(qdata)*0.8)
            n = 64

            trainingData = array(qdata[0:dataSplitPoint])
            validationData = array(qdata[dataSplitPoint:len(qdata)])
            print(trainingData.shape)
            print("start windowing")

            tdata2 = []
            for animation in trainingData:
                for subwindow in window(animation, n):
                    rangedData = array(subwindow).astype('float32')
                    tdata2.append(rangedData)
                    
            trainingData =  array(tdata2)

            valdata2 = []
            for animation in validationData:
                for subwindow in window(animation, n):
                    rangedData = array(subwindow).astype('float32')
                    valdata2.append(rangedData)

            validationData = array(valdata2)

            print("done windowing")
            originalShape = trainingData.shape
            print(originalShape)
            trainingData = trainingData.reshape(originalShape[0], originalShape[1], originalShape[2]*originalShape[3])

            validationData = validationData.reshape(validationData.shape[0], originalShape[1], originalShape[2]*originalShape[3])

            print(trainingData.shape)

            network = Sequential()
            degreesOFreedom = trainingData.shape[1] #joints * degreees of freedom
            windowSize = trainingData.shape[2] #temporal window 240 frames
            network.add(Dropout(0.25, input_shape=(degreesOFreedom, windowSize)))

            kernel_size = 16
            network.add(Conv1D(128, kernel_size, use_bias=True, activation='tanh', padding='same'))
            network.add(MaxPooling1D(padding='same'))
            
            network.add(Dropout(0.25))

            network.add(Conv1D(windowSize, kernel_size, use_bias=True, activation='tanh', padding='same'))

            network.add(UpSampling1D(size=2))

            network.summary()

            # load weights into new model
            network.load_weights("autoencoder_weights.h5")
            print("Loaded model from disk")

            network.compile(optimizer='adamax', loss='mse', metrics=['accuracy'])


            print(trainingData.shape)
            print(validationData.shape)
            network.fit(trainingData, trainingData, verbose=2,
                            epochs=200,
                            batch_size=1000,
                            shuffle=True,
                            validation_data=(validationData, validationData))
                            
            network.save_weights('autoencoder_weights.h5')

            print("finished")
            qdata = []
            trainingData = []
            validationData = []
            k = 0
            
print("finished")