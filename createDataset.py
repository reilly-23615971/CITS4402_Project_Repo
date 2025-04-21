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
Helper function to copy specified images to a new directory, renaming 
them in the process to avoid duplicate name issues

Parameters:
    images: list of image file names

    baseDir: directory where images are currently stored

    prefix: string to add to beginning of renamed files

    compilePath: folder to copy the images to
"""
def copyImageList(images, basePath, prefix, compilePath):
    for index, img in enumerate(images):
        _, ext = os.path.splitext(img)
        shutil.copy(
            os.path.join(basePath, img), 
            os.path.join(compilePath, (prefix + f'{index:05}' + ext))
        )

"""
Function to extract images from a given tarfile and select random ones
to form training and testing sets

Parameters:
    tarfilePath: name of (or path to) the tarfile containing the images 
    that will be used to make the training/testing sets

    trainSize: number of images to select for the training set

    imagePath: directory in tarfile to extract and select images from, 
    containing folders for positive and negative examples. Avoid 
    trailing slashes for this variable, as if one is present the code to
    select images from the tarfile may fail

    positiveSamples: folder in tarPath containing positive samples

    negativeSamples: folder in tarPath containing negative samples

    extractPath: directory to output the extracted tarfile images into

    trainPath: directory where the images selected for the training set 
    are outputted

    testPath: directory where the images selected for the testing set 
    are outputted

    withReplacement: whether duplicate images are possible in the sets

    randomState: NumPy seed for random selection reproducibility


Default parameters are based on the project task statement
Due to always selecting an equal number of positive and negative images 
for the training set, the total number of images in the set will be 1 
lower than trainSize when said parameter is odd
The testing set will always have 100 positive and 100 negative images 
per the project task statement
"""
def createDataset(
        tarfilePath, trainSize = 500, 
        extractPath = './Data/extractedData', imagePath = '', 
        positiveSamples = 'positive', negativeSamples = 'negative',
        trainPath = './Data/trainData', testPath = './Data/testData',
        withReplacement = False, randomSeed = None):
    # Validate numeric parameters
    if not isinstance(trainSize, int) or trainSize < 1:
        raise ValueError(
            f'trainSize should be a positive integer, was {trainSize}'
        )
    if not isinstance(randomSeed, int):
        raise ValueError(f'randomSeed should be an integer, was {randomSeed}')

    # Create output directories if they don't exist already
    for path in [extractPath, trainPath, testPath]:
        if not os.path.exists(path): os.makedirs(path)

    # Define possible image file extensions
    imageExts = {
        '.png', '.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', 
        '.webp', '.pgm', '.avif', '.gif', '.svg', '.bmp', '.tiff'
    }

    # Load the dataset
    with tarfile.open(tarfilePath, 'r') as tf:
        # Check that the specified folders are in the tarfile
        files = tf.getnames()
        if len([item for item in files if item.startswith(imagePath)]) == 0:
            raise FileNotFoundError((
                f'Specified image directory "{imagePath}" not found in '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePath, positiveSamples) not in files:
            raise FileNotFoundError((
                f'Specified positive sample directory "{positiveSamples}" not'
                f' found in image directory "{imagePath}" within '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePath, negativeSamples) not in files:
            raise FileNotFoundError((
                f'Specified negative sample directory "{negativeSamples}" not'
                f' found in image directory "{imagePath}" within '
                f'tarfile "{tarfilePath}"'
            ))
        
        # Extract just the images from the specified folder
        for file in tf.getmembers():
            filePath, fileExtension = os.path.splitext(file.name)
            if filePath.startswith(imagePath + '/') and fileExtension in imageExts: 
                tf.extract(file, path=extractPath)
    
    # TODO: Check if sampling files from the tarfile without extracting 
    # any other images is possible

    # Get locations of newly extracted positive and negative samples
    extractedPositive = os.path.join(extractPath, imagePath, positiveSamples)
    extractedNegative = os.path.join(extractPath, imagePath, negativeSamples)

    # Make sure enough samples were extracted
    positiveFiles = [
        img for img in os.listdir(extractedPositive) if os.path.isfile(
            os.path.join(extractPath, imagePath, positiveSamples, img)
        )
    ]
    negativeFiles = [
        img for img in os.listdir(extractedNegative) if os.path.isfile(
            os.path.join(extractPath, imagePath, negativeSamples, img)
        )
    ]
    if len(positiveFiles) == 0:
        raise FileNotFoundError(
            'No positive samples were extracted from the tarfile. Check'
            f' that the specified directory "{extractedPositive}" is '
            f'correct and that the specified tarfile "{tarfilePath}" '
            'contains the desired images.'
        )
    elif not withReplacement and len(positiveFiles) < (trainSize//2) + 100:
        raise ValueError(
            f'Training and testing datasets require {(trainSize//2) + 100} '
            f'positive samples total, but only {len(positiveFiles)} were '
            f'found. Check that the tarfile "{tarfilePath}" contains enough '
            'images, or set withReplacement = True to allow for duplicate '
            'samples in the datasets.'
        )
    elif len(negativeFiles) == 0:
        raise FileNotFoundError(
            'No negative samples were extracted from the tarfile. Check'
            f' that the specified directory "{extractedNegative}" is '
            f'correct and that the specified tarfile "{tarfilePath}" '
            'contains the desired images.'
        )
    elif not withReplacement and len(negativeFiles) < (trainSize//2) + 100:
        raise ValueError(
            f'Training and testing datasets require {(trainSize//2) + 100} '
            f'negative samples total, but only {len(negativeFiles)} were '
            f'found. Check that the tarfile "{tarfilePath}" contains enough '
            'images, or set withReplacement = True to allow for duplicate '
            'samples in the datasets.'
        )
    
    # Set random state for reproducibility
    rand = np.random.default_rng(randomSeed)

    # Randomly select enough files for training and testing sets
    sampledPositive = [img for img in rand.choice(
        os.listdir(extractedPositive), (trainSize//2) + 100, 
        replace = withReplacement
    )]
    sampledNegative = [img for img in rand.choice(
        os.listdir(extractedNegative), (trainSize//2) + 100, 
        replace = withReplacement
    )]

    # Split out training and testing images to ensure there's no overlap
    testPositive, trainPositive = sampledPositive[:100], sampledPositive[100:]
    testNegative, trainNegative = sampledNegative[:100], sampledNegative[100:]

    # Copy selected images into their own directories, changing the file
    # names to avoid similarly named positive and negative samples 
    # overwriting each other
    copyImageList(trainPositive, extractedPositive, 'p', trainPath)
    copyImageList(trainNegative, extractedNegative, 'n', trainPath)
    copyImageList(testPositive, extractedPositive, 'p', testPath)
    copyImageList(testNegative, extractedNegative, 'n', testPath)
    
    # TODO: Compress extracted datasets 

# Run createDataset for testing
createDataset(
    'DC-ped-dataset_base.tar', imagePath = '1', 
    positiveSamples = 'ped_examples', negativeSamples = 'non-ped_examples', 
    randomSeed = 42
)