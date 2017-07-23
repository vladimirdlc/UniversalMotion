from Quaternion import Quat
import bvh.reader
import bvh.helpers

from os import listdir
from os.path import isfile, join

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
            print(floats)
            for x,y,z in zip(*[iter(floats)]*3):
                quat = Quat((x,y,z))
                quaternionData.extend((quat.q+1)/2) #normalize the data

            framesQData.append(quaternionData)

        print(len(framesQData))
        print(len(framesQData[0]))
        #writeFrame(file, test.root, 0)
        #file.write("\n")
        file.close()


mypath = 'data/CMU/107/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for filename in onlyfiles:
    processBVH(mypath+filename)

#getFullFrame(file, test.root, 0)

