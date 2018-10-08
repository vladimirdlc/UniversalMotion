import os
import sys
import numpy as np
import scipy.io as io
import scipy.ndimage.filters as filters

sys.path.append('../../motion')
import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from itertools import islice


import heapq
""" hdm05 """
class_map = {
    'cartwheelLHandStart1Reps': 'cartwheel',
    'cartwheelLHandStart2Reps': 'cartwheel',
    'cartwheelRHandStart1Reps': 'cartwheel',
    'clap1Reps': 'clap',
    'clap5Reps': 'clap',
    'clapAboveHead1Reps': 'clap',
    'clapAboveHead5Reps': 'clap',
    # 'depositFloorR': 'deposit',
    # 'depositHighR': 'deposit',
    # 'depositLowR': 'deposit',
    # 'depositMiddleR': 'deposit',
    'depositFloorR': 'grab',
    'depositHighR': 'grab',
    'depositLowR': 'grab',
    'depositMiddleR': 'grab',
    'elbowToKnee1RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee1RepsRelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsLelbowStart': 'elbow_to_knee',
    'elbowToKnee3RepsRelbowStart': 'elbow_to_knee',
    'grabFloorR': 'grab',
    'grabHighR': 'grab',
    'grabLowR': 'grab',
    'grabMiddleR': 'grab',
    #'hitRHandHead': 'hit',
    #'hitRHandHead': 'grab',
    'hopBothLegs1hops': 'hop',
    'hopBothLegs2hops': 'hop',
    'hopBothLegs3hops': 'hop',
    'hopLLeg1hops': 'hop',
    'hopLLeg2hops': 'hop',
    'hopLLeg3hops': 'hop',
    'hopRLeg1hops': 'hop',
    'hopRLeg2hops': 'hop',
    'hopRLeg3hops': 'hop',
    'jogLeftCircle4StepsRstart': 'jog',
    'jogLeftCircle6StepsRstart': 'jog',
    'jogOnPlaceStartAir2StepsLStart': 'jog',
    'jogOnPlaceStartAir2StepsRStart': 'jog',
    'jogOnPlaceStartAir4StepsLStart': 'jog',
    'jogOnPlaceStartFloor2StepsRStart': 'jog',
    'jogOnPlaceStartFloor4StepsRStart': 'jog',
    'jogRightCircle4StepsLstart': 'jog',
    'jogRightCircle4StepsRstart': 'jog',
    'jogRightCircle6StepsLstart': 'jog',
    'jogRightCircle6StepsRstart': 'jog',
    'jumpDown': 'jump',
    'jumpingJack1Reps': 'jump',
    'jumpingJack3Reps': 'jump',
    'kickLFront1Reps': 'kick',
    'kickLFront2Reps': 'kick',
    'kickLSide1Reps': 'kick',
    'kickLSide2Reps': 'kick',
    'kickRFront1Reps': 'kick',
    'kickRFront2Reps': 'kick',
    'kickRSide1Reps': 'kick',
    'kickRSide2Reps': 'kick',
    'lieDownFloor': 'lie_down',
    'punchLFront1Reps': 'punch',
    'punchLFront2Reps': 'punch',
    'punchLSide1Reps': 'punch',
    'punchLSide2Reps': 'punch',
    'punchRFront1Reps': 'punch',
    'punchRFront2Reps': 'punch',
    'punchRSide1Reps': 'punch',
    'punchRSide2Reps': 'punch',
    'rotateArmsBothBackward1Reps': 'rotate_arms',
    'rotateArmsBothBackward3Reps': 'rotate_arms',
    'rotateArmsBothForward1Reps': 'rotate_arms',
    'rotateArmsBothForward3Reps': 'rotate_arms',
    'rotateArmsLBackward1Reps': 'rotate_arms',
    'rotateArmsLBackward3Reps': 'rotate_arms',
    'rotateArmsLForward1Reps': 'rotate_arms',
    'rotateArmsLForward3Reps': 'rotate_arms',
    'rotateArmsRBackward1Reps': 'rotate_arms',
    'rotateArmsRBackward3Reps': 'rotate_arms',
    'rotateArmsRForward1Reps': 'rotate_arms',
    'rotateArmsRForward3Reps': 'rotate_arms',
    # 'runOnPlaceStartAir2StepsLStart': 'run',
    # 'runOnPlaceStartAir2StepsRStart': 'run',
    # 'runOnPlaceStartAir4StepsLStart': 'run',
    # 'runOnPlaceStartFloor2StepsRStart': 'run',
    # 'runOnPlaceStartFloor4StepsRStart': 'run',
    'runOnPlaceStartAir2StepsLStart': 'jog',
    'runOnPlaceStartAir2StepsRStart': 'jog',
    'runOnPlaceStartAir4StepsLStart': 'jog',
    'runOnPlaceStartFloor2StepsRStart': 'jog',
    'runOnPlaceStartFloor4StepsRStart': 'jog',
    'shuffle2StepsLStart': 'shuffle',
    'shuffle2StepsRStart': 'shuffle',
    'shuffle4StepsLStart': 'shuffle',
    'shuffle4StepsRStart': 'shuffle',
    'sitDownChair': 'sit_down',
    'sitDownFloor': 'sit_down',
    'sitDownKneelTieShoes': 'sit_down',
    'sitDownTable': 'sit_down',
    'skier1RepsLstart': 'ski',
    'skier3RepsLstart': 'ski',
    'sneak2StepsLStart': 'sneak',
    'sneak2StepsRStart': 'sneak',
    'sneak4StepsLStart': 'sneak',
    'sneak4StepsRStart': 'sneak',
    'squat1Reps': 'squat',
    'squat3Reps': 'squat',
    'staircaseDown3Rstart': 'climb',
    'staircaseUp3Rstart': 'climb',
    'standUpKneelToStand': 'stand_up',
    'standUpLieFloor': 'stand_up',
    'standUpSitChair': 'stand_up',
    'standUpSitFloor': 'stand_up',
    'standUpSitTable': 'stand_up',
    'throwBasketball': 'throw',
    'throwFarR': 'throw',
    'throwSittingHighR': 'throw',
    'throwSittingLowR': 'throw',
    'throwStandingHighR': 'throw',
    'throwStandingLowR': 'throw',
    'turnLeft': 'turn',
    'turnRight': 'turn',
    'walk2StepsLstart': 'walk_forward',
    'walk2StepsRstart': 'walk_forward',
    'walk4StepsLstart': 'walk_forward',
    'walk4StepsRstart': 'walk_forward',
    'walkBackwards2StepsRstart': 'walk_backward',
    'walkBackwards4StepsRstart': 'walk_backward',
    'walkLeft2Steps': 'walk_left',
    'walkLeft3Steps': 'walk_left',
    'walkLeftCircle4StepsLstart': 'walk_left',
    'walkLeftCircle4StepsRstart': 'walk_left',
    'walkLeftCircle6StepsLstart': 'walk_left',
    'walkLeftCircle6StepsRstart': 'walk_left',
    'walkOnPlace2StepsLStart': 'walk_inplace',
    'walkOnPlace2StepsRStart': 'walk_inplace',
    'walkOnPlace4StepsLStart': 'walk_inplace',
    'walkOnPlace4StepsRStart': 'walk_inplace',
    'walkRightCircle4StepsLstart': 'walk_right',
    'walkRightCircle4StepsRstart': 'walk_right',
    'walkRightCircle6StepsLstart': 'walk_right',
    'walkRightCircle6StepsRstart': 'walk_right',
    'walkRightCrossFront2Steps': 'walk_right',
    'walkRightCrossFront3Steps': 'walk_right',
}

styletransfer_styles = [
    'angry', 'childlike', 'depressed', 'neutral', 
    'old', 'proud', 'sexy', 'strutting']
    
styletransfer_motions = [
    'fast_punching', 'fast_walking', 'jumping', 
    'kicking', 'normal_walking', 'punching', 
    'running', 'transitions']

class_names = list(sorted(list(set(class_map.values()))))

f = open('classes.txt', 'w')
f.write('\n'.join(class_names))
f.close()

def softmax(x, **kw):
    softness = kw.pop('softness', 1.0)
    maxi, mini = np.max(x, **kw), np.min(x, **kw)
    return maxi + np.log(softness + np.exp(mini - maxi))

def softmin(x, **kw):
    return -softmax(-x, **kw)

def quat2euler(joint):
    return Quaternions(joint).euler().ravel()
    
def euler2quat(joint):
    return Quaternions.from_euler(joint).ravel()
    
def normalizeForNN(joint):
    return (joint+1)/2
    
def denormalizeForNN(joint):
    #denormalizing quaternions data from [0, 1] to [-1,1]
    return (joint*2)-1
    
  
scale = 1000
    
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
    rotations = anim.rotations[:,1:len(anim.rotations)]

    print(len(rotations))
    """ Remove Uneeded Joints """
    reformatRotations = []

    #encoding
    for frame in rotations:
        joints = []
        
        for joint in frame:
            #print(joint)
            joints.append(joint*scale)

        reformatRotations.append(joints)
        
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


def get_files(directory):
    return [os.path.join(directory,f) for f in sorted(list(os.listdir(directory)))
    if os.path.isfile(os.path.join(directory,f))
    and f.endswith('.bvh') and f != 'rest.bvh'] 

#MSEConvertAndBackTest()

cmu_files = get_files('cmu')

cmu_rot_clips = []
for i, item in enumerate(cmu_files):
    print('Processing Rotation %i of %i (%s)' % (i, len(cmu_files), item))
    clips = process_file_rotations(item)
    cmu_rot_clips += clips
#   if i == 1: break


data_clips = np.array(cmu_rot_clips)

std = np.std(data_clips)
print(std)
mean = np.mean(data_clips)
data_clips -= mean
data_clips /= std
np.savez_compressed('cmu_rotations_Quat_cmu_20_standardized_w240_ws120_normalfps_scaled{}'.format(scale), clips=data_clips, std=std, mean=mean, scale=scale)

print(scale)
#np.savez_compressed('cmu_rotations_2samples_j30_w240_ws60_standardized_scaled10000', clips=data_clips, std=std, mean=mean, scale=scale)
#std = np.std(data_clips)
#mean = np.mean(data_clips)
#data_clips = (data_clips - mean) / std
#np.savez_compressed('cmu_rotations_full_30_normalizedwxyz', clips=data_clips, stdw=stdw, meanw=meanw, stdx=stdx, meanx=meanx, stdy=stdy, meany=meany, stdz=stdz, meanz=meanz)
#by channels
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