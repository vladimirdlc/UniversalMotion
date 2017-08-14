from Quaternion import Quat
from Quaternion import normalize
import bvh.reader
import bvh.helpers

from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense
from keras.models import Model
from numpy import array
import numpy as np

import math
import sys

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def writeFrame(file, node, i):
        if  node.position:
            for pos in node.position[i]:
                file.write("%s " % pos)
        
        for rot in node.rotation[i]:
            file.write("%s " % rot)

        for child in node.children:
            if len(child.channels) > 0:
                writeFrame(file, child, i)

def getFullFrame(node, i, fullFrame=[]):
        if  node.position:
            fullFrame.append(node.position[i])
        
        fullFrame.append(node.rotation[i])

        for child in node.children:
            if len(child.channels) > 0:
                getFullFrame(child, i, fullFrame)

        return fullFrame
        
def processBVH(fullPath):
        #test = bvh.reader.BvhReader(mypath+onlyfiles[0])
        #test.read();
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

        for currentFrame in data:
            quaternionData = []
            floats = [float(x) for x in currentFrame.split()]
            #print(floats)
            
            first = True
            
            for x,y,z in zip(*[iter(floats)]*3):
                if first:
                    rootPos.append((x,y,z))
                    first = False
                
                quat = Quat((x,y,z))
                #quaternionData.append(quat.q)
                if quat.q[3] < 0:
                    quaternionData.append(-quat.q)
                else:
                    quaternionData.append(quat.q)

                #quaternionData.append((sigmoid(x),sigmoid(y),sigmoid(z))) #normalize the data, extend?
                #quaternionData.append((x,y,z))

            framesQData.append(quaternionData)

        #print(len(framesQData))
        #print(len(framesQData[0]))
        #writeFrame(file, test.root, 0)
        #file.write("\n")
        file.close()
        
        return framesQData, rootPos

#sys.stdout=open("outputww.txt","w")

mypath = 'data/CMU/testing/'
onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

qdata = []
rootPos = []

for folder in onlyfolders:
    onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
    for filename in onlyfiles:
        qFrame, qRootPos = processBVH(mypath+folder+'/'+filename)
        qdata.extend(qFrame)
        rootPos.extend(qRootPos)

dataSplitPoint = int(len(qdata)*0.2)

trainingData = array(qdata).astype('float32')
#trainingData = array(qdata[dataSplitPoint:len(qdata)]).astype('float32')
#validationData = array(qdata[0:dataSplitPoint]).astype('float32')

trainingData = trainingData.reshape((len(trainingData), np.prod(trainingData.shape[1:])))
#validationData = validationData.reshape((len(validationData), np.prod(validationData.shape[1:])))

#trainingData = trainingData.T
#validationData = validationData.T

input_size = len(trainingData[0])
#encoding_dim = int(input_size*0.5) # compression of factor 20%, assuming the input is nn floats

# this is our input placeholder
input_frame = Input(shape=(input_size,))

# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_frame)
encoded = Dense(int(input_size-8), activation='tanh')(input_frame)
#encoded = Dense(int(input_size*0.6), activation='relu')(encoded)
#encoded = Dense(int(input_size*0.5), activation='relu')(encoded)

#decoded = Dense(int(input_size*0.6), activation='relu')(encoded)
#decoded = Dense(int(input_size*0.8), activation='relu')(decoded)

# "decoded" is the lossy reconstruction of the input
decoded = Dense(input_size, activation='tanh')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_frame, decoded)
autoencoder.summary()

#autoencoder.compile(optimizer='adadelta', loss='mse')
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# For a mean squared error regression problem
autoencoder.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

autoencoder.fit(trainingData, trainingData,
                epochs=100,
                batch_size=50,
                shuffle=True,
                validation_data=(trainingData, trainingData))

#                validation_split=0.2)
#model.fit(data, labels, validation_split=0.2)

#trainingData = array(qdata[len(qdata)-3698:len(qdata)]).astype('float32')
trainingData = array(qdata[0:3698]).astype('float32')

trainingData = trainingData.reshape((len(trainingData), np.prod(trainingData.shape[1:])))

decoded_quat = autoencoder.predict(trainingData)


file = open(mypath+'test.txt', 'w')

#for frame in decoded_quat:
#    for frameData in zip(*[iter(frame)]*input_size):
#        file.write('\n'+str(frameData))

#Denormalization
#decoded_quat = (decoded_quat*2)-1


i = 0
for frameData in decoded_quat:
    #for frameData in zip(*[iter(frame)]*input_size):
    file.write('\n')
    
    first = True
    for x,y,z,w in zip(*[iter(frameData)]*4):
        if first:
            file.write('{0} {1} {2} '.format(rootPos[i][0], rootPos[i][1], rootPos[i][2]))
            first = False
        else:
            quat = Quat(normalize((x,y,z,w)))
            file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
    
    i+=1

file.close()

#print(len(trainingData))
#print(len(validationData))

#(mypath+filename[1])
#getFullFrame(file, test.root, 0)

#exec(open("test.py").read())

print("finished")
#sys.stdout.close()
print("finished")

#print("finished")
