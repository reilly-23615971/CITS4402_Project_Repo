# CITS4402 Group Project
# Group Members:
# Felix Mavrodoglu (23720305)
# Jalil Inayat-Hussain (22751096)
# Reilly Evans (23615971)
# Functions used to generate datasets, extract features and fit models

# Imports
import os
import shutil
import tarfile
import numpy as np
import cv2
from sklearn.utils import shuffle
from skimage import io, color, transform
from skimage.feature import hog

"""
Function to extract images from a given tarfile, select a random sample
to form training and testing sets, and recompress the new sets into 
their own tarfiles


Parameters:
    tarfilePath: string containing the name of/path to the tarfile 
    containing the images that will be used to make the datasets

    trainSize: int representing the number of images to select for the 
    training set; defaults to 500 per the project task statement

    workingPath: string containing the name of/path to the temporary 
    folder where the tarfile's extracted images for the training/testing
    sets will go while the function is running. The function will raise 
    an error if this directory already exists; this is by design so as 
    to avoid overwriting existing folders

    imagePath: string containing the name of/path to the directory 
    within the tarfile to select and extract images from. This directory
    should contain separate folders for positive and negative examples

    guiPath: string containing the name of/path to the directory to 
    output images for testing the GUI

    positiveSamples: string containing the name of/path to the folder in
    imagePath containing positive image samples

    negativeSamples: string containing the name of/path to the folder in
    imagePath containing negative image samples

    trainOutput: string containing the name of/path to the file to 
    output the training set to, without any extensions

    testOutput: string containing the name of/path to the file to output
    the testing set to, without any extensions

    withReplacement: Boolean determining whether or not duplicate images
    are possible in the training and testing sets

    randomSeed: int representing the NumPy seed for ensuring random 
    selection is reproducible if necessary


Due to always selecting an equal number of positive and negative images 
for the training set, the total number of images in the set will be 1 
lower than trainSize when said parameter is odd
The testing set will always have 100 positive and 100 negative images in
order to follow the project task statement
"""
def createDataset(
        tarfilePath, trainSize = 500, 
        workingPath = './WorkingData',  imagePath = '', 
        guiPath = './Testing Images', positiveSamples = 'positive', 
        negativeSamples = 'negative', trainOutput = 'train_set', 
        testOutput = 'test_set', withReplacement = False, randomSeed = None):
    # Validate parameters
    if not isinstance(trainSize, int) or trainSize < 1:
        raise ValueError(
            f'trainSize should be a positive integer, was {trainSize}'
        )
    if not isinstance(randomSeed, int):
        raise ValueError(f'randomSeed should be an integer, was {randomSeed}')
    # Remove trailing slash from imagePath to prevent issues
    imagePathClean = imagePath if imagePath[-1] != '/' else imagePath[:-1]


    # Create working directories, throw error if they already exist
    try: os.mkdir(guiPath)
    except FileExistsError: raise FileExistsError((
        f'GUI test image directory "Testing Images" already exists; '
        'move or delete the existing directory before running again'
    ))
    try: os.mkdir(workingPath)
    except FileExistsError: raise FileExistsError((
        f'Specified working directory "{workingPath}" already exists; use a '
        'unique directory name instead'
    ))
    (trainPath, testPath) = (
        os.path.join(workingPath, trainOutput),
        os.path.join(workingPath, testOutput),
    )
    for path in [trainPath, testPath]: os.mkdir(path)

    # Define possible image file extensions
    imageExts = {
        '.png', '.jpg', '.jpeg', '.jfif', '.pjpeg', '.pjp', 
        '.webp', '.pgm', '.avif', '.gif', '.svg', '.bmp', '.tiff'
    }

    # Load the dataset
    with tarfile.open(tarfilePath, 'r') as tf:
        # Check that the specified folders are in the tarfile
        fileNames = tf.getnames()
        if len(
            [item for item in fileNames if item.startswith(imagePathClean)]
        ) == 0:
            # Delete the working directories since about to throw error
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise FileNotFoundError((
                f'Specified image directory "{imagePathClean}" not found in '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePathClean, positiveSamples) not in fileNames:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise FileNotFoundError((
                f'Specified positive sample directory "{positiveSamples}" not'
                f' found in image directory "{imagePathClean}" within '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePathClean, negativeSamples) not in fileNames:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise FileNotFoundError((
                f'Specified negative sample directory "{negativeSamples}" not'
                f' found in image directory "{imagePathClean}" within '
                f'tarfile "{tarfilePath}"'
            ))
        
        # Get the images in the specified folder
        files = []
        for file in tf.getmembers():
            filePath, fileExtension = os.path.splitext(file.name)
            if (filePath.startswith(imagePathClean + '/') and 
                fileExtension in imageExts): 
                files.append(file)

        # Make sure enough samples were found for each type
        positiveFiles = [
            img for img in files if img.name.startswith(
                os.path.join(imagePathClean, positiveSamples)
            )
        ]
        negativeFiles = [
            img for img in files if img.name.startswith(
                os.path.join(imagePathClean, negativeSamples)
            )
        ]
        if len(positiveFiles) == 0:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise FileNotFoundError((
                'No positive samples were found in the tarfile. '
                'Check that the specified directory '
                f'"{os.path.join(imagePathClean, positiveSamples)}" is '
                f'correct and that the specified tarfile "{tarfilePath}" '
                'contains the desired images.'
            ))
        elif not withReplacement and len(positiveFiles) < (trainSize//2) + 110:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise ValueError((
                f'Training and testing datasets require {(trainSize//2) + 110}'
                f' positive samples total, but only {len(positiveFiles)} were '
                f'found. Check that the tarfile "{tarfilePath}" contains '
                'enough images, or set withReplacement = True to allow for '
                'duplicate samples in the datasets.'
            ))
        elif len(negativeFiles) == 0:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise FileNotFoundError((
                'No negative samples were found in the tarfile. '
                'Check that the specified directory '
                f'"{os.path.join(imagePathClean, negativeSamples)}" is '
                f'correct and that the specified tarfile "{tarfilePath}" '
                'contains the desired images.'
            ))
        elif not withReplacement and len(negativeFiles) < (trainSize//2) + 110:
            shutil.rmtree(workingPath)
            shutil.rmtree(guiPath)
            raise ValueError((
                f'Training and testing datasets require {(trainSize//2) + 110}'
                f' negative samples total, but only {len(negativeFiles)} were '
                f'found. Check that the tarfile "{tarfilePath}" contains '
                'enough images, or set withReplacement = True to allow for '
                'duplicate samples in the datasets.'
            ))
        
        # Set random state for reproducibility
        rand = np.random.default_rng(randomSeed)
        
        # Randomly select enough files for training and testing sets
        sampledPositive = [img for img in rand.choice(
            positiveFiles, (trainSize//2) + 110, replace = withReplacement
        )]
        sampledNegative = [img for img in rand.choice(
            negativeFiles, (trainSize//2) + 110, replace = withReplacement
        )]

        # Split out training and testing images to ensure there's no 
        # overlap between the set images
        guiPositive = sampledPositive[:10]
        testPositive = sampledPositive[10:110]
        trainPositive = sampledPositive[110:]
        guiNegative = sampledNegative[:10]
        testNegative = sampledNegative[10:110]
        trainNegative = sampledNegative[110:]

        # Randomise the image order
        for set in (
            trainPositive, trainNegative,
            testPositive, testNegative, 
            guiPositive, guiNegative
        ): rand.shuffle(set)

        # Extract the sampled images, using new names to avoid similarly
        # named positive and negative samples overwriting each other
        # First letter of name indicates if sample is positive/negative
        for index, img in enumerate(trainPositive):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(trainPath, ('P' + f'{index:06}' + ext))
            )
        for index, img in enumerate(trainNegative):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(trainPath, ('N' + f'{index:06}' + ext))
            )
        for index, img in enumerate(testPositive):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(testPath, ('P' + f'{index:06}' + ext))
            ) 
        for index, img in enumerate(testNegative):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(testPath, ('N' + f'{index:06}' + ext))
            )
        for index, img in enumerate(guiPositive):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(guiPath, ('P' + f'{index:06}' + ext))
            ) 
        for index, img in enumerate(guiNegative):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(guiPath, ('N' + f'{index:06}' + ext))
            )
    
    # Compress the newly extracted datasets (not the GUI images)
    with tarfile.open(trainOutput + '.tar.gz', "w:gz") as tar:
        tar.add(trainPath, arcname = os.path.basename(trainPath))
    with tarfile.open(testOutput + '.tar.gz', "w:gz") as tar:
        tar.add(testPath, arcname = os.path.basename(testPath))

    # Delete the working directory now that sets have been compiled
    shutil.rmtree(workingPath)


"""
Function to compute the Histogram of Oriented Gradients (HOG) features 
on a given image with specific parameters


Parameters:
    imagePath: string containing the name of/path to the image whose 
    HOG features will be calculated

    numberOfOrientationBins: int representing the number of bins that 
    the gradient directions will be sorted into, a.k.a. the number of 
    unique orientations possible in the HOG features. Defaults to 9 bins
    per the project task statement

    cellDimensions: tuple containing 2 ints representing the length and 
    width of each cell used for HOG feature calculation, measured in 
    pixels. Defaults to (8, 8), i.e. 8*8 cells, per the project task 
    statement

    blockDimensions: tuple containing 2 ints representing the length and
    width of each block used for HOG feature calculation, measured in 
    cells. Defaults to (2, 2), i.e. 4 cells per block, per the project 
    task statement

    normalisationTechnique: string describing the normalisation method 
    used on the blocks. Possible values are 'L1' (the L1/Manhattan 
    norm), 'L1-sqrt' (the square root of the L1 norm), 'L2' (the 
    L2/Euclidean norm), and 'L2-Hys' (the L2 norm, then thresholding 
    high values to a maximum of 0.2, then the L2 norm again). Defaults 
    to 'L2-Hys' per the project task statement

    returnHOGImage: Boolean determining whether or not to return the HOG
    feature image visualisation alongside the features themselves. 
    Defaults to False


Outputs:
    features: 1D NumPy array containing the HOG features calculated on 
    the specified image

    hogImage: 2D NumPy array representing an image visualising the 
    calculated HOG features and their orientation. Only returned if 
    'returnHOGImage' is True

TODO: Check scikit-image hog source code to confirm it follows the 
report requirements in non-parameterised ways (gradient, block stride)
"""
def computeHOGFeatures(
        imagePath, numberOfOrientationBins = 9, cellDimensions = (8, 8),
        blockDimensions = (2, 2), normalisationTechnique = 'L2-Hys', 
        returnHOGImage = False):
    #TODO: input validation
    # Read the image
    img = io.imread(imagePath)
    
    # Resize to 64x128 pixels if needed
    if img.shape[0] != 128 or img.shape[1] != 64:
        img = transform.resize(img, (128, 64), anti_aliasing=True)
    
    # Convert image to grayscale (handle both RGB and RGBA images)
    if len(img.shape) > 2:
        # For RGBA images (4 channels), remove the alpha channel first
        if img.shape[2] == 4:
            img = img[:, :, :3]
        grayImg = color.rgb2gray(img)
    else:
        grayImg = img
    
    # Compute HOG features with specified parameters
    if returnHOGImage:
        features, hogImage = hog(
            grayImg,
            orientations = numberOfOrientationBins,
            pixels_per_cell = cellDimensions,
            cells_per_block = blockDimensions,
            block_norm = normalisationTechnique, 
            visualize = True,
            feature_vector = True,        # Return features as vector
            transform_sqrt = False,       # No gamma correction
        )
        return features, hogImage
    else:
        features = hog(
            grayImg,
            orientations = numberOfOrientationBins,
            pixels_per_cell = cellDimensions,
            cells_per_block = blockDimensions,
            block_norm = normalisationTechnique, 
            visualize = False,
            feature_vector = True,        # Return features as vector
            transform_sqrt = False,       # No gamma correction
        )
        return features



"""
Function to generate Histogram of Oriented Gradients (HOG) features and 
classifications for all images in a given tarfile and store them as a 
matrix of features alongside the classification (positive or negative) 
of each image
Uses computeHOGFeatures() above to calculate the HOG features


Parameters:
    tarfilePath: string containing the name of/path to the tarfile 
    containing the images that will be used to generate HOG features

    deleteDir: Boolean representing whether or not the directory 
    containing the extracted tarfile contents should be deleted once the
    function has completed

    randomSeed: int representing the NumPy seed for ensuring random 
    selection is reproducible if necessary

    numberOfOrientationBins: int representing the number of bins that 
    the gradient directions will be sorted into, a.k.a. the number of 
    unique orientations possible in the HOG features. Defaults to 9 bins
    per the project task statement. Originates from computeHOGFeatures()

    cellDimensions: tuple containing 2 ints representing the length and 
    width of each cell used for HOG feature calculation, measured in 
    pixels. Defaults to (8, 8), i.e. 8*8 cells, per the project task 
    statement. Originates from computeHOGFeatures()

    blockDimensions: tuple containing 2 ints representing the length and
    width of each block used for HOG feature calculation, measured in 
    cells. Defaults to (2, 2), i.e. 4 cells per block, per the project 
    task statement. Originates from computeHOGFeatures()

    normalisationTechnique: string describing the normalisation method 
    used on the blocks. Possible values are 'L1' (the L1/Manhattan 
    norm), 'L1-sqrt' (the square root of the L1 norm), 'L2' (the 
    L2/Euclidean norm), and 'L2-Hys' (the L2 norm, then thresholding 
    high values to a maximum of 0.2, then the L2 norm again). Defaults 
    to 'L2-Hys' per the project task statement. Originates from 
    computeHOGFeatures()


Outputs:
    imagePath: 1D NumPy array of strings corresponding to the file paths
    to each image in the dataset. Useful as an index value for the data

    imageFeatures: Nested Numpy array of numeric values representing the
    HOG features generated from each image

    imageClass: 1D NumPy array of Booleans indicating each image was a 
    positive/human sample (True) or a negative/non-human sample (False),
    determined via the file name
"""
def formatDataset(tarfilePath, deleteDir = True, randomSeed = None, 
        numberOfOrientationBins = 9, cellDimensions = (8, 8),
        blockDimensions = (2, 2), normalisationTechnique = 'L2-Hys'):
    #TODO: input validation
    # Extract the training set
    with tarfile.open(tarfilePath, 'r:gz') as tar:
        # Extract images to directory
        imageDir = os.path.join('.', tar.getmembers()[0].name)
        tar.extractall()
        # Get list of image file paths
        imagePath = np.array([
            os.path.join('.', img.name) for img in tar.getmembers()[1:]
        ])
    # Compute HOG features
    imageFeatures = np.array([
        computeHOGFeatures(
            img, numberOfOrientationBins, cellDimensions, blockDimensions, 
            normalisationTechnique, returnHOGImage = False
        )[1] for img in imagePath
    ])
    # Get classification label of each image
    imageClass = np.char.startswith(
        imagePath, os.path.join(imageDir, 'P')
    )
    # Shuffle dataset
    imagePath, imageFeatures, imageClass = shuffle(
        imagePath, imageFeatures, imageClass, random_state=randomSeed
    )
    # Delete extracted directory before returning (if specified)
    if deleteDir: shutil.rmtree(imageDir)
    return imagePath, imageFeatures, imageClass



'''
# Run createDataset to generate Daimler datasets
createDataset(
    'DC-ped-dataset_base.tar', imagePath = '1', 
    positiveSamples = 'ped_examples', negativeSamples = 'non-ped_examples', 
    randomSeed = 42
)
'''