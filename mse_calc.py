import re
from numpy import *
from random import uniform
import argparse as Ap

set_printoptions(suppress=True)

def getArgParser():
    parser = Ap.ArgumentParser(description='Parameters to add noise')

    parser.add_argument("--fileA", default="input_144_21.bvh", type=str)
    parser.add_argument("--fileB", default="144_21_decoded_RotatationMatrix_cmu_rotations_RotatationMatrix_cmu_21_standardized_w240_ws120_normalfps_scaled1.bvh", type=str)
    parser.add_argument("--skiprows", default="129", type=int)
    args = parser.parse_args()
    return args

def angleDiff( angle1, angle2 , ignoreFirstRot = True):
    startidx = 6 if ignoreFirstRot else 0

    angle1 = (angle1 + 180)
    angle2 = (angle2 + 180)

    diff = angle1

    i = 0
    for line in diff:
        j = 0

        for c in line:
            if j < startidx:
                j +=1
                continue

            diff[i][j] = min(abs(angle1[i][j]-angle2[i][j]),abs(angle2[i][j]-angle1[i][j]))
            j += 1
        i += 1

    return diff

def mseAngle(a, b):
    return square(angleDiff(a, b)).mean()

#args = getArgParser()

#dataA = loadtxt(args.fileA, skiprows=args.skiprows)
#dataB = loadtxt(args.fileB, skiprows=args.skiprows)

#print(args.fileA)
#print(args.fileB)

def mseAB_files(fileA, fileB, skiprows=129):
    dataA = loadtxt(fileA, skiprows=skiprows)
    dataB = loadtxt(fileB, skiprows=skiprows)

    return mseAngle(dataA, dataB)

