import os
import sys
import numpy as np
from enum import Enum
import re
from random import uniform
import argparse as Ap


class Decoder(Enum):
    QUATERNION = 'Quaternion'
    ROTATION_MATRIX = 'RotatationMatrix'
    EULER = 'Euler'
    AXIS_ANGLE = 'AxisAngle'

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from itertools import islice
import eulerangles as eang

wdw = 240
step = 120
scale = 10000000

decodeType = Decoder.AXIS_ANGLE

def getArgParser():
    parser = Ap.ArgumentParser(description='Parameters to add noise')
    parser.add_argument("--file", default="numbers_array.txt", type=str)
    parser.add_argument("--ofile", default="numbers_array_out.txt", type=str)
    parser.add_argument("--le", default="-90", type=float)
    parser.add_argument("--ue", default="90", type=float)
    parser.add_argument("--llimit", default="-180", type=float)
    parser.add_argument("--ulimit", default="180", type=float)
    parser.add_argument("--ignoreFirstRotation", default="True", type=bool)
    parser.add_argument("--noisy", default="True", type=bool)

    args = parser.parse_args()
    return args


args = getArgParser()

def process_file_rotations(filename, window=240, window_step=120):
    anim, names, frametime = BVH.load(filename, order='zyx')

    """ Convert to 60 fps """
    # anim = anim[::2]

    '''anim = anim[:,np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]

    names = np.array(names)
    names = names[np.array([
         0,
         2,  3,  4,  5,
         7,  8,  9, 10,
        12, 13, 15, 16,
        18, 19, 20, 22,
        25, 26, 27, 29])]

    print(names.shape)
    filename = filename.replace('.bvh', '_')
    BVH.save(filename+'parsed.bvh', anim)
	'''

    """ Do FK """
    print(len(anim.rotations))

    """ Remove Uneeded Joints """
    # exported
    rotations = anim.rotations[:, 0:len(anim.rotations)]

    """ Remove Uneeded Joints """
    reformatRotations = []

    # encoding
    for frame in rotations:
        joints = []
        first = True

        for joint in frame:
            if args.noisy:
                if first:
                    first = False
                else:
                    x, y, z = Quaternions(joint).euler().ravel()
                    joint = [addNoise(x), addNoise(y), addNoise(z)]
                    joint = Quaternions.from_euler(np.array(joint), order='xyz').ravel()

            if decodeType is Decoder.QUATERNION:
                joints.append(joint * scale)
            elif decodeType is Decoder.EULER:
                joints.append(Quaternions(joint).euler().ravel() * scale)
            elif decodeType is Decoder.AXIS_ANGLE:
                angle, axis = Quaternions(joint).angle_axis()
                input = axis.flatten()
                input = np.insert(input, 0, angle)
                input = np.array(input)  # 4 values
                joints.append(input * scale)
            elif decodeType is Decoder.ROTATION_MATRIX:
                euler = Quaternions(joint).euler().ravel()  # we get x,y,z
                # eang library uses convention z,y,x
                m = eang.euler2mat(euler[2], euler[1], euler[0])
                input = np.array(m[0].tolist() + m[1].tolist() + m[2].tolist())  # 9 values
                joints.append(input*scale)

        reformatRotations.append(joints)

    rotations = np.array(reformatRotations)

    print(rotations.shape)

    """ Slide over windows """
    windows = []

    for j in range(0, len(rotations) - window // 8, window_step):
        """ If slice too small pad out by repeating start and end poses """
        slice = rotations[j:j + window]

        if len(slice) < window:
            left = slice[:1].repeat((window - len(slice)) // 2 + (window - len(slice)) % 2, axis=0)
            right = slice[-1:].repeat((window - len(slice)) // 2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)

        if len(slice) != window: raise Exception()

        windows.append(slice)

    return windows

def get_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']


def addNoise(qw):
    frand = uniform(args.le, args.ue) + qw

    if frand > args.ulimit:
        return frand - 360
    elif frand < args.llimit:
        return frand + 360

    return frand


cmu_files = get_files('cmu')
cmu_rot_clips = []

wdw = 240
step = 120

for i, item in enumerate(cmu_files):
    print('Processing Rotation %i of %i (%s)' % (i, len(cmu_files), item))
    clips = process_file_rotations(item, window=wdw, window_step=step)
    #if len(clips) < wdw:
    #    continue

    cmu_rot_clips += clips
    #data_clips = np.array(cmu_rot_clips)
    # if i == 1: break

data_clips = np.array(cmu_rot_clips)
print(data_clips.shape)

std = np.std(data_clips)
print(std)
mean = np.mean(data_clips)
data_clips -= mean
data_clips /= std

if args.noisy:
    np.savez_compressed('cmu_rotations_{}_cmu_{}_standardized_w{}_ws{}_normalfps_scaled{}_noisy'.format(decodeType.value, data_clips.shape[2], wdw, step, scale), clips=data_clips, std=std, mean=mean, scale=scale)
else:
    np.savez_compressed('cmu_rotations_{}_cmu_{}_standardized_w{}_ws{}_normalfps_scaled{}'.format(decodeType.value, data_clips.shape[2], wdw, step, scale), clips=data_clips, std=std, mean=mean, scale=scale)

print(scale)
