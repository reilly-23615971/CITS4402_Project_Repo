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
Function to extract images from a given tarfile, select a random sample
to form training and testing sets, and recompress the new sets into 
their own tarfiles


Parameters:
    tarfilePath: name of (or path to) the tarfile containing the images 
    that will be used to make the training/testing sets

    trainSize: number of images to select for the training set; defaults
    to 500 since that is the minimum set by the project task statement

    workingPath: the temporary folder where the tarfile's extracted 
    images for the training/testing sets will go while the function is 
    executing. The function will raise an error if this directory 
    already exists; this is by design so as to avoid overwriting 
    existing folders.

    imagePath: directory within the tarfile to select and extract images
    from, containing folders for positive and negative examples.

    positiveSamples: folder in imagePath containing positive samples

    negativeSamples: folder in imagePath containing negative samples

    trainOutput: name of the file to output the training set to, without
    any extensions or paths

    testOutput: name of the file to output the testing set to, without 
    any extensions or paths

    withReplacement: whether or not duplicate images are possible in the
    training and testing sets

    randomSeed: NumPy seed for random selection reproducibility


Due to always selecting an equal number of positive and negative images 
for the training set, the total number of images in the set will be 1 
lower than trainSize when said parameter is odd
The testing set will always have 100 positive and 100 negative images in
order to follow the project task statement
"""
def createDataset(
        tarfilePath, trainSize = 500, 
        workingPath = './WorkingData',  imagePath = '', 
        positiveSamples = 'positive', negativeSamples = 'negative',
        trainOutput = 'train_set', testOutput = 'test_set',
        withReplacement = False, randomSeed = None):
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
            # Delete the working directory since about to throw error
            shutil.rmtree(workingPath)
            raise FileNotFoundError((
                f'Specified image directory "{imagePathClean}" not found in '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePathClean, positiveSamples) not in fileNames:
            shutil.rmtree(workingPath)
            raise FileNotFoundError((
                f'Specified positive sample directory "{positiveSamples}" not'
                f' found in image directory "{imagePathClean}" within '
                f'tarfile "{tarfilePath}"'
            ))
        elif os.path.join(imagePathClean, negativeSamples) not in fileNames:
            shutil.rmtree(workingPath)
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
            raise FileNotFoundError((
                'No positive samples were found in the tarfile. '
                'Check that the specified directory '
                f'"{os.path.join(imagePathClean, positiveSamples)}" is '
                f'correct and that the specified tarfile "{tarfilePath}" '
                'contains the desired images.'
            ))
        elif not withReplacement and len(positiveFiles) < (trainSize//2) + 100:
            shutil.rmtree(workingPath)
            raise ValueError((
                f'Training and testing datasets require {(trainSize//2) + 100}'
                f' positive samples total, but only {len(positiveFiles)} were '
                f'found. Check that the tarfile "{tarfilePath}" contains '
                'enough images, or set withReplacement = True to allow for '
                'duplicate samples in the datasets.'
            ))
        elif len(negativeFiles) == 0:
            shutil.rmtree(workingPath)
            raise FileNotFoundError((
                'No negative samples were found in the tarfile. '
                'Check that the specified directory '
                f'"{os.path.join(imagePathClean, negativeSamples)}" is '
                f'correct and that the specified tarfile "{tarfilePath}" '
                'contains the desired images.'
            ))
        elif not withReplacement and len(negativeFiles) < (trainSize//2) + 100:
            shutil.rmtree(workingPath)
            raise ValueError((
                f'Training and testing datasets require {(trainSize//2) + 100}'
                f' negative samples total, but only {len(negativeFiles)} were '
                f'found. Check that the tarfile "{tarfilePath}" contains '
                'enough images, or set withReplacement = True to allow for '
                'duplicate samples in the datasets.'
            ))
        
        # Set random state for reproducibility
        rand = np.random.default_rng(randomSeed)
        
        # Randomly select enough files for training and testing sets
        sampledPositive = [img for img in rand.choice(
            positiveFiles, (trainSize//2) + 100, replace = withReplacement
        )]
        sampledNegative = [img for img in rand.choice(
            negativeFiles, (trainSize//2) + 100, replace = withReplacement
        )]

        # Split out training and testing images to ensure there's no 
        # overlap between the set images
        testPositive = sampledPositive[:100]
        trainPositive = sampledPositive[100:]
        testNegative = sampledNegative[:100]
        trainNegative = sampledNegative[100:]

        # Extract the sampled images, using new names to avoid similarly
        # named positive and negative samples overwriting each other
        for index, img in enumerate(trainPositive + trainNegative):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(trainPath, (f'{index:05}' + ext))
            )
        for index, img in enumerate(testPositive + testNegative):
            _, ext = os.path.splitext(img.name)
            tf._extract_member(
                img, os.path.join(testPath, (f'{index:05}' + ext))
            ) 
    
    # Compress the newly extracted datasets
    with tarfile.open(trainOutput + '.tar.gz', "w:gz") as tar:
        tar.add(trainPath, arcname = os.path.basename(trainPath))
    with tarfile.open(testOutput + '.tar.gz', "w:gz") as tar:
        tar.add(testPath, arcname = os.path.basename(testPath))

    # Delete the working directory now that sets have been compiled
    shutil.rmtree(workingPath)

# Run createDataset for testing
createDataset(
    'DC-ped-dataset_base.tar', imagePath = '1', 
    positiveSamples = 'ped_examples', negativeSamples = 'non-ped_examples', 
    randomSeed = 42
)