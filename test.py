from Quaternion import Quat
from Quaternion import normalize
import bvh.reader
import bvh.helpers

from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential, model_from_json
from numpy import array
import numpy as np

import math
import sys

from itertools import islice


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
        rootRot = []

        for currentFrame in data:
            quaternionData = []
            floats = [float(x) for x in currentFrame.split()]
            #print(floats)
            
            first = True
            second = True
            
            for x,y,z in zip(*[iter(floats)]*3):
                if first:
                    rootPos.append((x,y,z))
                    first = False
                elif second:
                    rootRot.append((x,y,z))
                    second = False
                
                quat = Quat((x,y,z))
                #quaternionData.append(quat.q)
                #if quat.q[3] < 0:
                #    quaternionData.append(-quat.q)
                #else:
                quaternionData.append(quat.q)

                #quaternionData.append((sigmoid(x),sigmoid(y),sigmoid(z))) #normalize the data, extend?
                #quaternionData.append((x,y,z))

            framesQData.append(quaternionData)
            
        #print(len(framesQData))
        #print(len(framesQData[0]))
        #writeFrame(file, test.root, 0)
        #file.write("\n")
        file.close()
        
        return framesQData, rootPos, rootRot

#sys.stdout=open("outputww.txt","w")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

mypath = 'data/CMU/testing/'
onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

qdata = []
rootPos = []
rootRot = []

'''

targetFrames = 25


for folder in onlyfolders:
    onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
    for filename in onlyfiles:
        currentFrameChunk = 0;
        qFrame, qRootPos, qRootRot = processBVH(mypath+folder+'/'+filename)
        
        for subwindow in window(qFrame, targetFrames): #window
            qdata.append(subwindow)
            
        for subwindow in window(qRootPos, targetFrames):
            rootPos.append(subwindow)
        
        for subwindow in window(qRootRot, targetFrames):
            rootRot.append(subwindow)

dataSplitPoint = int(len(qdata)*0.8)

trainingData = array(qdata).astype('float32')
validationData = array(qdata[dataSplitPoint:len(qdata)]).astype('float32')
trainingData = array(qdata[0:dataSplitPoint]).astype('float32')

n = 240


#trainingData = trainingData.reshape((len(trainingData), np.prod(trainingData.shape[1:])))
#validationData = validationData.reshape((len(validationData), np.prod(validationData.shape[1:])))


#s[0]
#trainingData.shape

firstInputSize = len(trainingData[0])

tdata2 = []
for subwindow in window(trainingData, n): #window
    rangedData = array(subwindow).astype('float32')
    tdata2.append(rangedData.flatten())

trainingData =  array(tdata2)

valdata2 = []
for subwindow in window(validationData, n):
    rangedData = array(subwindow).astype('float32')
    valdata2.append(rangedData.flatten())

validationData = array(valdata2)


    
#trainingData = trainingData.T
#validationData = validationData.T

input_size = len(trainingData[0])
#encoding_dim = int(input_size*0.5) # compression of factor 20%, assuming the input is nn floats



X = np.load('data_rotation_full_cmu.npz')['clips']

print(X.shape)

#X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
#X = np.reshape(X, )
X = np.swapaxes(X, 1, 2).astype('float32')

trainingData = X
validationData = X

print(X.shape)

network = Sequential()
#https://keras.io/layers/convolutional/#conv1d
#(17978, 84, 240)
degreesOFreedom = X.shape[1] #84
window = 240
batch_size = 1
network.add(Dropout(0.25, input_shape=(degreesOFreedom, window)))
#3*22 joints + 4 contact timings + 2 root translational velocity + 1 root rotational velocity
#4*22

batchsize=1
kernel_size = 25
network.add(Conv1D(256, kernel_size, use_bias=True, activation='tanh', padding='same'))
network.add(MaxPooling1D(padding='same'))
#now the decoder
#network.add(UpSampling1D(size=2), output_shape=(batchsize, 256, window))
network.add(UpSampling1D(size=2))
network.add(Dropout(0.25))

network.add(Conv1D(240, kernel_size, use_bias=True, activation='tanh', padding='same'))

network.summary()
#(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
#model.add(Dropout(0.25))
# "encoded" is the encoded representation of the input
#encoded = Dense(encoding_dim, activation='relu')(input_frame)
#autoencoder.compile(optimizer='adadelta', loss='mse')
#autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
#autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
# For a mean squared error regression problem
network.compile(optimizer='rmsprop', loss='mse', metrics=['accuracy'])

model_json = network.to_json()
with open("network.json", "w") as json_file:
    json_file.write(model_json)


print(trainingData.shape)
print(validationData.shape)
network.fit(trainingData, trainingData, verbose=1,
                epochs=200,
                batch_size=50,
                shuffle=True,
                validation_data=(validationData, validationData))

network.save_weights('autoencoder_weights.h5')
'''

# load json and create model
json_file = open('network.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
network = model_from_json(loaded_model_json)
# load weights into new model
network.load_weights("autoencoder_weights.h5")
print("Loaded model from disk")

#trainingData = array(qdata[len(qdata)-3698:len(qdata)]).astype('float32')
trainingData = array(qdata[0:3690]).astype('float32').flatten() #remember to put a multiple of 15

tdata2 = []
for toDecodeData in chunks(trainingData, n*firstInputSize):
    rangedData = array(toDecodeData).astype('float32')
    tdata2.append(rangedData.flatten())

trainingData =  array(tdata2)



decoded_quat = autoencoder.predict(trainingData)

decData = []

for partition in decoded_quat:
        for g10array in zip(*[iter(partition)]*firstInputSize):
            decData.append(g10array)

decoded_quat = array(decData)

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
    second = True
    
    for x,y,z,w in zip(*[iter(frameData)]*4):
        if first:
            file.write('{0} {1} {2} '.format(rootPos[i][0], rootPos[i][1], rootPos[i][2]))
            first = False
        elif second:
            quat = Quat(normalize((x,y,z,w)))
            xo = (rootRot[i][0])
            yo = (rootRot[i][1])
            zo = (rootRot[i][2])
            
            file.write('{0} {1} {2} '.format(xo, yo, zo))
            second = False
        else:
            quat = Quat(normalize((x,y,z,w)))
            file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
    
    i+=1

file.close()
#DO HALF OF ROTATION POSITION AVG TARGET WITH MODEL


#print(len(trainingData))
#print(len(validationData))

#(mypath+filename[1])
#getFullFrame(file, test.root, 0)

#exec(open("test.py").read())

#sys.stdout.close()
print("finished")

#print("finished")