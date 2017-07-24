from Quaternion import Quat
import bvh.reader
import bvh.helpers

from os import listdir
from os.path import isfile, join

from keras.layers import Input, Dense
from keras.models import Model
from numpy import array

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

        for currentFrame in data:
            quaternionData = []
            floats = [float(x) for x in currentFrame.split()]
            #print(floats)
            
            for x,y,z in zip(*[iter(floats)]*3):
                quat = Quat((x,y,z))
                quaternionData.extend((quat.q+1)/2) #normalize the data

            framesQData.append(quaternionData)

        print(len(framesQData))
        print(len(framesQData[0]))
        #writeFrame(file, test.root, 0)
        #file.write("\n")
        file.close()
        
        return framesQData

encoding_dim = 80  # 88 floats -> compression of factor 50%, assuming the input is 176 floats

# this is our input placeholder
input_frame = Input(shape=(176,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_frame)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(176, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_frame, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_frame, encoded)

# create a placeholder for an encoded (176-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

mypath = 'data/CMU/testing/'
onlyfolders = [f for f in listdir(mypath) if not isfile(join(mypath, f))]

qdata = []

for folder in onlyfolders:
    onlyfiles = [f for f in listdir(mypath+folder) if isfile(join(mypath+folder, f))]
    for filename in onlyfiles:
        qdata.extend(processBVH(mypath+folder+'/'+filename))

dataSplitPoint = int(len(qdata)*0.2)
trainingData = array(qdata[dataSplitPoint:len(qdata)])
validationData = array(qdata[0:dataSplitPoint])

#trainingd = []
#for tdata in trainingData:
#    trainingd.extend(tdata)

#validationd = []
#for vdata in validationData:
#    validationd.extend(vdata)

#vdata = tdata
    
#x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
#trainingData = trainingData.reshape((len(trainingData), np.prod(trainingData.shape[1:])))
#validationData = validationData.reshape((len(validationData), np.prod(validationData.shape[1:])))

autoencoder.fit(trainingData, trainingData,
                epochs=50,
                batch_size=256,
                shuffle=False,
                validation_data=(validationData, validationData))
                
print(len(trainingData))
print(len(validationData))

#(mypath+filename[1])
#getFullFrame(file, test.root, 0)

#exec(open("test.py").read())