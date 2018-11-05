import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from mse_calc import mseAB_files
from keras.models import load_model
import eulerangles as eang
from numpy import array
import numpy as np

import sys

sys.path.insert(0, './motion')

import BVH as BVH
from Quaternions import Quaternions
from enum import Enum

class Decoder(Enum):
    QUATERNION = 'Quaternion'
    ROTATION_MATRIX = 'RotationMatrix'
    EULER = 'Euler'
    AXIS_ANGLE = 'AxisAngle'

def mse(a, b):
    return np.square(np.subtract(a, b)).mean()


def normalize(array):
    """
    Normalize a 4 element array/list/numpy.array for use as a quaternion

    :param quat_array: 4 element list/array
    :returns: normalized array
    :rtype: numpy array
    """
    quat = np.array(array)
    return quat / np.sqrt(np.dot(quat, quat))

def process_file_rotations(filename, window=240, window_step=120):
    anim, names, frametime = BVH.load(filename, order='zyx')

    """ Convert to 60 fps """
    # anim = anim[::2]

    """ Do FK """
    print(len(anim.rotations))

    """ Remove Uneeded Joints """
    # exported
    rotations = anim.rotations[:, 0:len(anim.rotations)]

    print(len(rotations))
    """ Remove Uneeded Joints """
    reformatRotations = []

    # encoding
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

        reformatRotations.append(joints)

    rotations = np.array(reformatRotations)
    rotations = rotations.reshape(rotations.shape[0], rotations.shape[1]*rotations.shape[2])

    #    (2448, 21, 4) -> (2448, 84)
    print(rotations.shape)

    """ Slide over windows """
    windows = []
    partition = []

    for rot in rotations:
        partition.append(rot)
        if len(partition) >= window:
            windows.append(partition)
            partition = []

    return windows

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

outputFolder = 'decoded/'
#filename = '144_21_parsed'

allDecodes = [Decoder.AXIS_ANGLE, Decoder.EULER, Decoder.QUATERNION, Decoder.ROTATION_MATRIX]
#'144_21', '144_21_45d', '144_21_90d', 'gorilla_run', 'gorilla_run_45d', 'gorilla_run_90d', 'gorilla_run_asymmetric',
allFiles = ['144_21', '144_21_45d', '144_21_90d', 'gorilla_run', 'gorilla_run_45d', 'gorilla_run_90d', 'gorilla_run_asymmetric',
            'b0041_kicking', 'b0041_kicking_45d', 'b0041_kicking_90d']
#allFiles = ['144_21']

customFrameTime = 0.031667
file1 = open("MSE.txt", "a")

for filename in allFiles:
    fullPathAnim = 'data/decoding/' + filename + '.bvh'

    print('processing...')

    np.set_printoptions(suppress=True)

    for decodeType in allDecodes:
        fileToDecode = 'cmu_rotations_'+decodeType.value+'_cmu_21_standardized_w240_ws120_normalfps_scaled1'
        X = np.load(fileToDecode+'.npz')
        mean = X['mean']
        std = X['std']
        scale = X['scale']

        print('\n')

        X = np.array(np.load(fileToDecode+'.npz')['clips'])

        np.random.seed(0)
        # split into 80% for train and 20% for tests

        network = load_model(
            'models/'+fileToDecode+'_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_model.h5')
        network.compile(optimizer='adam', loss='mse')
        network.summary()


        network.load_weights(
            'weights/'+fileToDecode+'_k25_hu256_vtq3_e600_d0.15_bz128_valtest0.2_activationrelu_weights.h5')

        print('decoding...')

        #print(">MSE I/O NN Q <> QHat:")
        #print(mse(trainingData, decoded_quat))
        folder60fps = outputFolder+'60fps/'+filename+'/'
        if not os.path.exists(folder60fps):
            os.makedirs(folder60fps)

        anim, names, frametime = BVH.load(fullPathAnim, order='zyx', world=False)

        BVH.save(outputFolder+'input_'+filename+'.bvh', anim)

        anim60 = anim[::2] # convert to 60 fps
        BVH.save(outputFolder+'60fps/'+filename+'/'+'input_'+filename+'.bvh', anim60, frametime=customFrameTime)


        # globalRot = anim.rotations[:,0:1]
        rotations = anim.rotations[:, 0:len(anim.rotations)]  # 1:len(anim.rotations) to avoid glogal rotation

        print(len(rotations))
        """ Remove Uneeded Joints """
        reformatRotations = []

        # encoding
        for frame in rotations:
            joints = []

            for joint in frame:
                if decodeType is Decoder.QUATERNION:
                    joints.append(joint)
                elif decodeType is Decoder.EULER:
                    joints.append(Quaternions(joint).euler().ravel())
                elif decodeType is Decoder.AXIS_ANGLE:
                    angle, axis = Quaternions(joint).angle_axis()
                    input = axis.flatten()
                    input = np.insert(input, 0, angle)
                    input = np.array(input) #4 values
                    joints.append(input)
                elif decodeType is Decoder.ROTATION_MATRIX:
                    euler = Quaternions(joint).euler().ravel() #we get x,y,z
                    #eang library uses convention z,y,x
                    m = eang.euler2mat(euler[2], euler[1], euler[0])
                    input = np.array(m[0].tolist()+m[1].tolist()+m[2].tolist()) #9 values
                    joints.append(input)

            reformatRotations.append(joints*scale)

        reformatRotations = np.array(reformatRotations)

        datatypeLength = X.shape[3] #4 for quaternions

        X = process_file_rotations(fullPathAnim, window=X.shape[1], window_step=X.shape[1])
        X = np.array(list(X))
        print(X.shape)
        X -= mean
        X /= std
        #X = X.reshape(X.shape[0], X.shape[1], X.shape[2] * X.shape[3])

        decoded_quat = array(network.predict(X))

        rotations = (((decoded_quat)*std)+mean)/scale

        idx = 0

        #go by all windows 240
        for wdw in rotations:
            for frame in wdw:
                if idx >= anim.rotations.shape[0]:
                    break

                skipFirst = True
                j = 0
                frameLine = []
                for joint in chunks(frame, datatypeLength):
                    if skipFirst is True:
                        j+=1
                        skipFirst = False
                        continue

                    if decodeType is Decoder.QUATERNION:
                        anim.rotations[idx][j] = Quaternions(joint)
                    elif decodeType is Decoder.EULER:
                        joint = [joint[2], joint[1], joint[0]]
                        anim.rotations[idx][j] = Quaternions.from_euler(np.array(joint), order='zyx')
                    elif decodeType is Decoder.AXIS_ANGLE:
                        z, y, x = eang.angle_axis2euler(joint[0], [joint[1], joint[2], joint[3]]) #theta, x, y, z
                        joint = np.degrees([z, y, x])  # in z,y,x format
                        joints.append(joint)
                    elif decodeType is Decoder.ROTATION_MATRIX:
                        m0 = np.array([joint[0], joint[1], joint[2]])
                        m1 = np.array([joint[3], joint[4], joint[5]])
                        m2 = np.array([joint[6], joint[7], joint[8]])
                        m = [m0, m1, m2]
                        joint = eang.mat2euler(m)  # in z,y,x rad format
                        joints.append(joint)
                    j += 1
                idx += 1

        fullFileName = outputFolder+filename+'_decoded_'+decodeType.value+'_'+fileToDecode+'.bvh'
        BVH.save(fullFileName, anim)
        fullFileName = outputFolder+'60fps/'+filename+'/'+filename+'_decoded_'+decodeType.value+'_'+fileToDecode+'.bvh'
        anim60 = anim[::2] # convert to 60fps
        BVH.save(fullFileName, anim60, frametime=customFrameTime)

        #calculating mse
        fileA = outputFolder + '60fps/' + filename + '/' + 'input_' + filename + '.bvh'
        fileB = fullFileName

        total_mse = mseAB_files(fileA.replace("_90d","").replace("_45d",""), fileB)
        strout = ("f1:{},f2:{},{}\n".format(fileA, fileB, "{0:.4f}".format(total_mse)))
        print(strout)

        file1.write(strout)

        print("finished "+fullFileName)

file1.close()