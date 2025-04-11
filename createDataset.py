# CITS4402 Group Project
# Group Members:
# Felix Mavrodoglu (23720305)
# Jalil Inayat-Hussain (22751096)
# Reilly Evans (23615971)
# Code for generating the dataset used by our SVM classifier

# Imports
import os
import shutil
import tarfile
import numpy as np

# Constants
trainSize = 500
extractLocation = os.path.abspath('./Data/extractedData')
testOutput = os.path.abspath('./Data/testData')
trainOutput = os.path.abspath('./Data/trainData')

# Load the Daimer dataset
with tarfile.open("DC-ped-dataset_base.tar", "r") as tf:
    tf.extract(member='1/non-ped_examples', path="./test")
    for file in tf.getmembers():
        if file.name.startswith('1/'): tf.extract(file, path=extractLocation)

# TODO: Check if sampling from the tarfile without extracting the whole thing is possible

# Select files for training and testing; use 50/50 split of pedestrian and non-pedestrian for training
pedestrianDir = os.path.join(extractLocation, '1/ped_examples')
nonPedestrianDir = os.path.join(extractLocation, '1/non-ped_examples')
testPedestrian = [img for img in np.random.choice(os.listdir(pedestrianDir), 100, replace=False)]
testNonPedestrian = [img for img in np.random.choice(os.listdir(nonPedestrianDir), 100, replace=False)]
trainPedestrian = [img for img in np.random.choice(os.listdir(pedestrianDir), trainSize//2, replace=False)]
trainNonPedestrian = [img for img in np.random.choice(os.listdir(nonPedestrianDir), trainSize//2, replace=False)]

# Copy selected images to their own directory
# TODO: Check to ensure that like-numbered images from pedestrian and non-pedestrian folders don't override each other
if not os.path.exists(testOutput):
    os.makedirs(testOutput)
if not os.path.exists(trainOutput):
    os.makedirs(trainOutput)
for img in testPedestrian:
    imgPath = os.path.join(pedestrianDir, img)
    shutil.copy(imgPath, testOutput)
for img in testNonPedestrian:
    imgPath = os.path.join(nonPedestrianDir, img)
    shutil.copy(imgPath, testOutput)
for img in trainPedestrian:
    imgPath = os.path.join(pedestrianDir, img)
    shutil.copy(imgPath, trainOutput)
for img in trainNonPedestrian:
    imgPath = os.path.join(nonPedestrianDir, img)
    shutil.copy(imgPath, trainOutput)

# TODO: Compress extracted datasets