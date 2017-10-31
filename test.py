from Quaternion import Quat
from Quaternion import normalize
from Quaternions import Quaternions #fix
from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv1D, Dropout, MaxPooling1D, UpSampling1D
from keras.models import Model, Sequential, model_from_json
from sklearn.model_selection import train_test_split
from numpy import array
import numpy as np


import math
from math import radians, degrees
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
        
def window_stack(a, stepsize=1, width=3):
    n = a.shape[0]
    return np.hstack( a[i:1+n+i-width:stepsize] for i in range(0,width) )

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
                else:
                    #quaternionData.append((x,y,z))
                    #quat = quaternion.from_euler_angles(x, y, z)
                    #quat = quaternion.as_float_array(quaternion.from_euler_angles(x,y,z))[0]
                    quat = Quat((x,y,z))
                    #quaternionData.append(quat.q)
                    if quat.q[3] < 0:
                        #quaternionData.append(-quat.q
                        quaternionData.append(-quat.q)
                    else:
                        #quaternionData.append(quat.q)
                        quaternionData.append(quat.q)
                        

                    #quat = quaternion.as_euler_angles(np.quaternion(quat.q[0], quat[1], quat[2], quat[3]))
                    
                    #print('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
                
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

'''
def writeDecodedToFile(decodedArray, file):
    i = 0
    print("writing to file")
    sizeDecoded = len(decodedArray)
    
    for windowData in decodedArray:
        i += 1
        print("{} of {} ".format(i,sizeDecoded))
        
        second = True
        for frame in windowData:
            first = True
            file.write('\n')
            for x,y,z,w in zip(*[iter(frame)]*4):
                if first:
                    file.write('{0} {1} {2} '.format(0, 0, 0))
                    eulers = Quaternions(np.array([x, y, z, w])).normalized().euler().flatten()
                    #print(eulers[0])
                    eulers = np.rad2deg(eulers)
                    file.write('{0} {1} {2} '.format(eulers[0], eulers[1], eulers[2]))
                    
                    #file.write('{0} {1} {2} '.format(rootPos[i][0], rootPos[i][1], rootPos[i][2]))
                    first = False
                else:
                    #quat = Quat(normalize((x,y,z,w)))
                    eulers = Quaternions(np.array([x, y, z, w])).normalized().euler().flatten()
                    eulers = np.rad2deg(eulers)
                    file.write('{0} {1} {2} '.format(eulers[0], eulers[1], eulers[2]))
                    #file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))

'''

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
                #quat = Quat(normalize((x,y,z,w)))
                if first:
                #    file2.write('{} {} {} {} {} {} {} {} '.format(0, 0, 0, 0, 0, 0, 0, 0))
                     file.write('{0} {1} {2} {3} {4} {5} '.format(0, 0, 0, 0, 0, 0))
                     first = False
                
                #qx = np.array((x,y,z,w))
                #qx = qx / np.sqrt(np.dot(qx, qx))
                #quat = Quat(normalize(qx))
                #file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))0
                #qx = np.array((x,y,z,w))
                #qx = qx / np.sqrt(np.dot(qx, qx))
                quat = Quat((x,y,z, w))
                #quat = quaternion.as_euler_angles(np.quaternion(qx[0], qx[1], qx[2], qx[3]))[-]
                #quat = quaternion.as_euler_angles(np.quaternion(x, y, z, w))
                file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
                #file.write('{0} {1} {2} '.format(degrees(quat[0]), degrees(quat[1]), degrees(quat[2])))
                
                file2.write('{0} {1} {2} {3} '.format(x, y, z, w))


    
mypath = 'data/CMU/testing/'
onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

qdata = []
#rootPos = []
#rootRot = []

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


targetFrames = 25


for folder in onlyfolders:
    onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
    k = 0
    for filename in onlyfiles:


        qFrame, qRootPos, qRootRot = processBVH(mypath+folder+'/'+filename)
        
        #for subwindow in window(qFrame, targetFrames): #window
        qdata.append(array(qFrame))
            
        #for subwindow in window(qRootPos, targetFrames):
        #rootPos.append(qRootPos)
        
        #for subwindow in window(qRootRot, targetFrames):
        #rootRot.append(qRootRot)
        k += 1;
        
        if k == 8:
        
            qdata = array(qdata)
            dataSplitPoint = int(len(qdata)*0.8)

            n = 64


            #trainingData = trainingData.reshape((len(trainingData), np.prod(trainingData.shape[1:])))
            #validationData = validationData.reshape((len(validationData), np.prod(validationData.shape[1:])))


            #s[0]
            #trainingData.shape

            trainingData = array(qdata[0:dataSplitPoint])
            validationData = array(qdata[dataSplitPoint:len(qdata)])
            print(trainingData.shape)
            print("start windowing")

            tdata2 = []
            for animation in trainingData:
                #for subwindow in window(animation, n): #window
                #windows = array(np.split(array(animation),n))
                for subwindow in window(animation, n):
                    rangedData = array(subwindow).astype('float32')
                    tdata2.append(rangedData)
                    
            trainingData =  array(tdata2)

            valdata2 = []
            for animation in validationData:
                #windows = array(np.split(array(animation),n))
                for subwindow in window(animation, n):
                    rangedData = array(subwindow).astype('float32')
                    valdata2.append(rangedData)

            validationData = array(valdata2)

            print("done windowing")
            originalShape = trainingData.shape
            print(originalShape)
            trainingData = trainingData.reshape(originalShape[0], originalShape[1], originalShape[2]*originalShape[3])
            #trainingData = np.swapaxes(trainingData, 1, 2).astype('float32')

            validationData = validationData.reshape(validationData.shape[0], originalShape[1], originalShape[2]*originalShape[3])
            #validationData = np.swapaxes(validationData, 1, 2).astype('float32')

                
            #trainingData = trainingData.T
            #validationData = validationData.T

            #encoding_dim = int(input_size*0.5) # compression of factor 20%, assuming the input is nn floats

            '''

            X = np.load('data_rotation_full_cmu.npz')['clips']

            initialXShape = X.shape
            print(X.shape)

            #X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)
            X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3])
            #X = np.reshape(X, )
            X = np.swapaxes(X, 1, 2).astype('float32')

            #trainingData = X
            #validationData = X

            dataSplitPoint = int(len(X)*0.8)

            valData = X[dataSplitPoint:len(X)]
            trainingData = X[0:dataSplitPoint]

            dataSplitPoint2 = int((len(valData)-100)) #making it small enough

            validationData = valData[0:dataSplitPoint2]

            #trainingData, validationData = train_test_split(X, test_size=0.2)

            print(X.shape)

            '''

            print(trainingData.shape)

            network = Sequential()
            degreesOFreedom = trainingData.shape[1] #84 
            windowSize = trainingData.shape[2] #240 frames
            batch_size = 1
            network.add(Dropout(0.25, input_shape=(degreesOFreedom, windowSize)))

            kernel_size = 9
            network.add(Conv1D(128, kernel_size, use_bias=True, activation='tanh', padding='same'))
            network.add(MaxPooling1D(padding='same'))
            
            network.add(Conv1D(windowSize, kernel_size, use_bias=True, activation='tanh', padding='same'))

            network.add(UpSampling1D(size=2))
            network.add(Dropout(0.25))

            network.summary()
            #(filters, kernel_size, strides=1, padding='valid', dilation_rate=1, activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
            #model.add(Dropout(0.25))
            # "encoded" is the encoded representation of the input
            #encoded = Dense(encoding_dim, activation='relu')(input_frame)
            #autoencoder.compile(optimizer='adadelta', loss='mse')
            #autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
            #autoencoder.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            # For a mean squared error regression problem

            #model_json = network.to_json()
            #with open("network.json", "w") as json_file:
            #    json_file.write(model_json)

            # load json and create model
            #json_file = open('network.json', 'r')
            #loaded_model_json = json_file.read()
            #json_file.close()
            #network = model_from_json(loaded_model_json)

            # load weights into new model
            network.load_weights("autoencoder_weights.h5")
            print("Loaded model from disk")

            network.compile(optimizer='adam', loss='mse', metrics=['accuracy'])


            print(trainingData.shape)
            print(validationData.shape)
            network.fit(trainingData, trainingData, verbose=2,
                            epochs=200,
                            batch_size=1000,
                            shuffle=True,
                            validation_data=(validationData, validationData))
                            
            network.save_weights('autoencoder_weights.h5')


            #trainingData = array(qdata[len(qdata)-3698:len(qdata)]).astype('float32')
            #trainingData = array(qdata[0:3690]).astype('float32').flatten() #remember to put a multiple of 15

            #testingData = validationData[len(validationData)-1000:len(validationData)]
            #decData = []

            #for partition in decoded_quat:
            #        for g10array in zip(*[iter(partition)]*firstInputSize):
            #            decData.append(g10array)

            #decoded_quat = array(decData)


            #for frame in decoded_quat:
            #    for frameData in zip(*[iter(frame)]*input_size):
            #        file.write('\n'+str(frameData))

            #Denormalization
            #decoded_quat = (decoded_quat*2)-1

            #X = np.concatenate([Xcmu, Xhdm05, Xmhad, Xstyletransfer, Xedin_locomotion, Xedin_xsens, Xedin_misc, Xedin_punching], axis=0)

            #X = np.reshape(X, )

            '''
            testingData = trainingData[0:40]
            #testingData = np.swapaxes(testingData, 1, 2).astype('float32')

            decoded_quat = network.predict(testingData)

            file = open(mypath+'test.txt', 'w')
            file2 = open(mypath+'test2.txt', 'w')

            #testingData = np.swapaxes(testingData, 1, 2).astype('float32')

            #testingData = np.reshape(testingData, (testingData.shape[0], originalShape[1], originalShape[2], originalShape[3]))

            writeDecodedToFile(testingData, file, file2)
            #testingData = testingData.reshape(testingData.shape[0], initialXShape[1], initialXShape[2], initialXShape[3])
            file.write("--------------------")
            file2.write("--------------------")
            #decoded_quat = np.swapaxes(array(decoded_quat), 1, 2).astype('float32')
            #decoded_quat = np.swapaxes(array(decoded_quat), 1, 2).astype('float32')
            #writeDecodedToFile(decoded_quat, file)


            i = 0
            print("writing to file 2")
            sizeDecoded = len(decoded_quat)

            for windowData in decoded_quat:
                for frame in windowData:
                    file.write('\n')
                    file2.write('\n')
                    first = True
                    second = True
                    
                    for x,y,z,w in zip(*[iter(frame)]*4):
                        if first:
                            #file.write('{0} {1} {2} '.format(rootPos[i][0], rootPos[i][1], rootPos[i][2]))
                            file.write('{0} {1} {2} '.format(0,0,0))
                            file.write('{0} {1} {2} '.format(0,0,0))
                            file2.write('{0} {1} {2} {3} '.format(0,0,0,1))
                            file2.write('{0} {1} {2} {3} '.format(0,0,0,1))
                            first = False
                        #elif second:
                            #quat = Quat(normalize((x,y,z,w)))
                            #xo = (rootRot[i][0])
                            #yo = (rootRot[i][1])
                            #zo = (rootRot[i][2])
                            
                            #file.write('{0} {1} {2} '.format(xo, yo, zo))
                            second = False
                        #print(x,y,z,w)
                        qx = np.array((x,y,z,w))
                        qx = qx / np.sqrt(np.dot(qx, qx))
                        
                        quat = Quat(normalize(qx))
                        #quat = quaternion.as_euler_angles(np.quaternion(qx[0], qx[1], qx[2], qx[3]))
                        #quat = quaternion.as_euler_angles(np.quaternion(x, y, z, w))
                        #file.write('{0} {1} {2} '.format(degrees(quat[0]), degrees(quat[1]), degrees(quat[2])))
                        
                        file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
                        #file.write('{0} {1} {2} '.format(quat.ra, quat.dec, quat.roll))
                        file2.write('{0} {1} {2} {3}'.format(x,y,z,w))
                    
                    i+=1
                    print("{} of {} ".format(i,sizeDecoded))

            file.close()
            
            '''
            #DO HALF OF ROTATION POSITION AVG TARGET WITH MODEL


            #print(len(trainingData))
            #print(len(validationData))

            #(mypath+filename[1])
            #getFullFrame(file, test.root, 0)

            #exec(open("test.py").read())


            #sys.stdout.close()
            print("finished")
            qdata = []
            trainingData = []
            validationData = []
            k = 0
#print("finished")