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


"""
Function to extract images from a given tarfile and select random ones
to form training and testing sets

Parameters:
    tarfilePath: name of (or path to) the tarfile containing the images 
    that will be used to make the training/testing sets

    trainSize: number of images to select for the training set

    imagePath: directory in tarfile to select images from, containing 
    folders for positive and negative examples. Must end in '/' to allow
    for extracting only the images from the tarfile

    positiveSamples: folder in tarPath containing positive samples

    negativeSamples: folder in tarPath containing negative samples

    extractPath: directory to output the extracted tarfile images into

    trainPath: directory where the images selected for the training set 
    are outputted

    testPath: directory where the images selected for the testing set 
    are outputted

    withReplacement: whether duplicate images are possible in the sets

    randomState: NumPy seed for random selection reproducibility


Default parameters are based on the Daimer dataset and the project 
task statement where applicable
Due to always selecting an equal number of positive and negative images,
the number of images in the training dataset will be 1 lower than 
trainSize when the parameter is odd
The testing set will always have 100 positive and 100 negative images 
per the project task statement
"""
def createDataset(
        tarfilePath, trainSize = 500, 
        extractPath = './Data/extractedData', imagePath = '1/', 
        positiveSamples = 'ped_examples', negativeSamples = 'non-ped_examples',
        trainPath = './Data/testData', testPath = './Data/trainData',
        withReplacement = False, randomState = None):
    # TODO: Validate parameters

    # Create output directories if they don't exist already
    for path in [extractPath, trainPath, testPath]:
        if not os.path.exists(path): os.makedirs(path)

    # Load the dataset
    with tarfile.open(tarfilePath, 'r') as tf:
        for file in tf.getmembers():
            if file.name.startswith(imagePath): 
                tf.extract(file, path=extractPath)
    
    # TODO: Check if sampling from the tarfile without extracting the whole thing is possible

    # Get locations of newly extracted positive and negative samples
    extractedPositive = os.path.join(extractPath, imagePath, positiveSamples)
    extractedNegative = os.path.join(extractPath, imagePath, negativeSamples)

    # Set random state for reproducibility
    np.random.seed(randomState)

    # Randomly select enough files for training and testing sets
    sampledPositive = [img for img in np.random.choice(
        os.listdir(extractedPositive), (trainSize//2) + 100, 
        replace = withReplacement
    )]
    sampledNegative = [img for img in np.random.choice(
        os.listdir(extractedNegative), (trainSize//2) + 100, 
        replace = withReplacement
    )]

    # Split out training and testing images to ensure there's no overlap
    testPositive, trainPositive = sampledPositive[:100], sampledPositive[100:]
    testNegative, trainNegative = sampledNegative[:100], sampledNegative[100:]

    # TODO: Add prefix to positive/negative images so that no identical image issues occur

    # Copy selected images into their own directories
    for img in trainPositive:
        newPath = os.path.join(extractedPositive, img)
        shutil.copy(newPath, trainPath)
    for img in trainNegative:
        newPath = os.path.join(extractedNegative, img)
        shutil.copy(newPath, trainPath)
    for img in testPositive:
        newPath = os.path.join(extractedPositive, img)
        shutil.copy(newPath, testPath)
    for img in testNegative:
        newPath = os.path.join(extractedNegative, img)
        shutil.copy(newPath, testPath)
    
    # TODO: Compress extracted datasets 

# Run createDataset
createDataset('DC-ped-dataset_base.tar', randomState = 42)