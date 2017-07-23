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


mypath = 'data/CMU/102/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

test = bvh.reader.BvhReader(mypath+onlyfiles[0])
test.read();
file = open('test.txt', 'w')

print(getFullFrame(test.root, 0))
writeFrame(file, test.root, 0)
file.write("\n")
file.close()

#getFullFrame(file, test.root, 0)

