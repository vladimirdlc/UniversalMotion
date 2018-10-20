import re
from numpy import *
from random import uniform
import argparse as Ap

set_printoptions(suppress=True)

def getArgParser():
    parser = Ap.ArgumentParser(description='Parameters to add noise')

    parser.add_argument("--file", default="numbers_array.txt", type=str)
    parser.add_argument("--ofile", default="numbers_array_out.txt", type=str)
    parser.add_argument("--le", default="-90", type=float)
    parser.add_argument("--ue", default="90", type=float)
    parser.add_argument("--llimit", default="-180", type=float)
    parser.add_argument("--ulimit", default="180", type=float)
    parser.add_argument("--ignorePosition", default="True", type=bool)

    args = parser.parse_args()
    return args


def addNoise(qw):
    frand = uniform(args.le, args.ue) + qw

    if frand > args.ulimit:
        return frand - 360
    elif frand < args.llimit:
        return frand + 360

    return frand

args = getArgParser()
data = loadtxt(args.file)

print(data[0][1])
print(data.shape)

startidx = 6 if args.ignoreFirstRotation else 3

for i in range(0, data.shape[0]):
    for j in range(startidx, data.shape[1]):
        data[i][j] = addNoise(data[i][j])

savetxt(args.ofile, data, fmt='%f')