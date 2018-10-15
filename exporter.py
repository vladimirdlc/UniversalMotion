import os
import sys
import numpy as np
from enum import Enum

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from itertools import islice
import eulerangles as eang


wdw = 240
step = 120
scale = 1
    
def process_file_rotations(filename, window=240, window_step=120):
    anim, names, frametime = BVH.load(filename, order='zyx')

    """ Convert to 60 fps """
    #anim = anim[::2]
    
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
	#exported
    rotations = anim.rotations[:,0:len(anim.rotations)]

    print(len(rotations))
    """ Remove Uneeded Joints """
    reformatRotations = []

    #encoding
    for frame in rotations:
        joints = []

        for joint in frame:
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
                joints.append(input)

        
    rotations = np.array(reformatRotations)

    print(rotations.shape)

    """ Slide over windows """
    windows = []
    windows_classes = []
    
    for j in range(0, len(rotations)-window//8, window_step):
        print(j)
        #input(j)
        """ If slice too small pad out by repeating start and end poses """
        slice = rotations[j:j+window]

        if len(slice) < window:
            left  = slice[:1].repeat((window-len(slice))//2 + (window-len(slice))%2, axis=0)
            right = slice[-1:].repeat((window-len(slice))//2, axis=0)
            slice = np.concatenate([left, slice, right], axis=0)
        
        if len(slice) != window: raise Exception()
        
        windows.append(slice)

    return windows


class Decoder(Enum):
    QUATERNION = 'Quaternion'
    ROTATION_MATRIX = 'Ro'
    EULER = 3
    AXIS_ANGLE = 4

def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 


decodeType = Decoder.QUATERNION
#MSEConvertAndBackTest()

cmu_files = get_files('cmu')
cmu_rot_clips = []

for i, item in enumerate(cmu_files):
    print('Processing Rotation %i of %i (%s)' % (i, len(cmu_files), item))
    clips = process_file_rotations(item, window=wdw, window_step=step)
    cmu_rot_clips += clips

data_clips = np.array(cmu_rot_clips)
print(data_clips.shape)

std = np.std(data_clips)
print(std)
mean = np.mean(data_clips)
data_clips -= mean
data_clips /= std
np.savez_compressed('cmu_rotations_Quat_cmu_{}_standardized_w{}_ws{}_normalfps_scaled{}'.format(data_clips.shape[2], wdw, step, scale), filesinfo=filesidx, clips=data_clips, std=std, mean=mean, scale=scale)
#np.savez_compressed('cmu_rotations_Quat_cmu_20_standardized_w480_ws240_normalfps_scaled{}'.format(scale), filesinfo=filesidx, clips=data_clips, std=std, mean=mean, scale=scale)

print(scale)
'''
cmu_rot_clips = []
for i, item in enumerate(cmu_files):
    print('Processing Rotation %i of %i (%s)' % (i, len(cmu_files), item))
    clips = process_file_rotations(item)
    cmu_rot_clips += clips
data_clips = np.array(cmu_rot_clips)
np.savez_compressed('data_rotation_cmu_quat_30', clips=data_clips)


cmu_clips = []
for i, item in enumerate(cmu_files):
    print('Processing %i of %i (%s)' % (i, len(cmu_files), item))
    clips, _ = process_file(item)
    cmu_clips += clips
data_clips = np.array(cmu_clips)
np.savez_compressed('data_cmu', clips=data_clips)

hdm05_files = get_files('hdm05')
hdm05_clips = []
hdm05_classes = []
for i, item in enumerate(hdm05_files):
    print('Processing %i of %i (%s)' % (i, len(hdm05_files), item))
    clips, cls = process_file(item)
    hdm05_clips += clips
    hdm05_classes += cls    
data_clips = np.array(hdm05_clips)
data_classes = np.array(hdm05_classes)
np.savez_compressed('data_hdm05', clips=data_clips, classes=data_classes)

"""
styletransfer_files = get_files('styletransfer')
styletransfer_clips = []
styletransfer_classes = []
for i, item in enumerate(styletransfer_files):
    print('Processing %i of %i (%s)' % (i, len(styletransfer_files), item))
    clips, cls = process_file(item)
    styletransfer_clips += clips
    styletransfer_classes += cls    
data_clips = np.array(styletransfer_clips)
np.savez_compressed('data_styletransfer', clips=data_clips, classes=styletransfer_classes)
"""

edin_locomotion_files = get_files('edin_locomotion')
edin_locomotion_clips = []
for i, item in enumerate(edin_locomotion_files):
    print('Processing %i of %i (%s)' % (i, len(edin_locomotion_files), item))
    clips, _ = process_file(item)
    edin_locomotion_clips += clips    
data_clips = np.array(edin_locomotion_clips)
np.savez_compressed('data_edin_locomotion', clips=data_clips)

edin_xsens_files = get_files('edin_xsens')
edin_xsens_clips = []
for i, item in enumerate(edin_xsens_files):
    print('Processing %i of %i (%s)' % (i, len(edin_xsens_files), item))
    clips, _ = process_file(item)
    edin_xsens_clips += clips    
data_clips = np.array(edin_xsens_clips)
np.savez_compressed('data_edin_xsens', clips=data_clips)

edin_kinect_files = get_files('edin_kinect')
edin_kinect_clips = []
for i, item in enumerate(edin_kinect_files):
    print('Processing %i of %i (%s)' % (i, len(edin_kinect_files), item))
    clips, _ = process_file(item)
    edin_kinect_clips += clips
data_clips = np.array(edin_kinect_clips)
np.savez_compressed('data_edin_kinect', clips=data_clips)

edin_misc_files = get_files('edin_misc')
edin_misc_clips = []
for i, item in enumerate(edin_misc_files):
    print('Processing %i of %i (%s)' % (i, len(edin_misc_files), item))
    clips, _ = process_file(item)
    edin_misc_clips += clips
data_clips = np.array(edin_misc_clips)
np.savez_compressed('data_edin_misc', clips=data_clips)

mhad_files = get_files('mhad')
mhad_clips = []
for i, item in enumerate(mhad_files):
    print('Processing %i of %i (%s)' % (i, len(mhad_files), item))
    clips, _ = process_file(item)
    mhad_clips += clips    
data_clips = np.array(mhad_clips)
np.savez_compressed('data_mhad', clips=data_clips)

edin_punching_files = get_files('edin_punching')
edin_punching_clips = []
for i, item in enumerate(edin_punching_files):
    print('Processing %i of %i (%s)' % (i, len(edin_punching_files), item))
    clips, _ = process_file(item)
    edin_punching_clips += clips
data_clips = np.array(edin_punching_clips)
np.savez_compressed('data_edin_punching', clips=data_clips)

edin_terrain_files = get_files('edin_terrain')
edin_terrain_clips = []
for i, item in enumerate(edin_terrain_files):
    print('Processing %i of %i (%s)' % (i, len(edin_terrain_files), item))
    clips, _ = process_file(item)
    edin_terrain_clips += clips
data_clips = np.array(edin_terrain_clips)
np.savez_compressed('data_edin_terrain', clips=data_clips)
'''