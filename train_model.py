
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from projectFunctions import formatDataset
import joblib
import shutil


def train_and_save_model():
    train_tar_path = './ExampleSets/train_set.tar.gz'
    imagePath, imageFeatures, imageClass = formatDataset(
    tarfilePath=train_tar_path,
    deleteDir=True,           
    randomSeed=42              )
    model = LinearSVC(random_state=42)
    model.fit(imageFeatures, imageClass)
    joblib.dump(model, "svm_model.joblib")



print("SVM model trained and saved to 'svm_model.joblib'")



