from sklearn.preprocessing import LabelEncoder
import copy
from sklearn.preprocessing import MinMaxScaler
from skimage import feature
from matplotlib import pyplot as plt
import numpy as np
import mahotas
import cv2
import os
import h5py

import glob
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
import imutils
import warnings


warnings.filterwarnings('ignore')

# parameters
images_per_class = {
    "circinatum": 53,
    "garryana": 70,
    "glabrum": 62,
    "kelloggii": 80,
    "macrophyllum": 67,
    "negundo": 31
}

images_for_test = [13,14,13,17,15,7]

fixed_size       = tuple((512, 512))
train_path       = "dataset/train"
test_path        = "dataset/test"
output_path      = "output"

LBPradius        = 3
LBPpoints        = 8 * LBPradius

zernikeRadius    = 150

num_trees = 100
test_size = 0.10
seed      = 9
scoring    = "accuracy"

# Hu Moments (shape)
def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

# Haralick Texture (texture)
def fd_haralick(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick

# Color Histogram (color)
def fd_histogram(image, mask=None):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

# Local Binary Patterns (texture)
def fd_localBinaryPatterns(image):
    eps = 1e-7
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, LBPpoints, LBPradius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0,LBPpoints + 3), range=(0, LBPpoints + 2))
    hist = hist.astype("float")
    eps=1e-7
    hist /= (hist.sum() + eps)
    return hist.flatten()

# Zernike Moments (shape)
def fd_zernikeMoments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return mahotas.features.zernike_moments(image, zernikeRadius)

# Histogram of Oriented Gradients
def fd_histogramOrientedGradients(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
            cells_per_block=(2, 2), transform_sqrt=True, block_norm="L1")
    return H.flatten()

# Oriented FAST and Rotated BRIEF (local feature)
def fd_orb(image):
    orb = cv2.ORB_create(nfeatures = 100)
    keypoints = orb.detect(image,None)
    (_, descriptors) = orb.compute(image, keypoints)
    ft=descriptors.flatten()
    if len(ft)<100:
        ft = np.zeros(100-len(ft),dtype=int)
    else:
        ft = np.array(ft[0:100])
    return ft

# get the training labels
train_labels = os.listdir(train_path)

# sort the training labels
train_labels.sort()
print(train_labels)

# empty lists to hold feature vectors and labels
global_features = []
labels          = []

# loop over the training data sub-folders
for training_name in train_labels:
    # join the training data path and each species training folder
    dir = os.path.join(train_path, training_name)

    # get the current training label
    current_label = training_name

    # loop over the images in each sub-folder
    for x in range(1,images_per_class[current_label]+1):
        # get the image file name
        if x < 10:
            file = dir + "/l0" + str(x) + ".jpg"
        else:
            file = dir + "/l" + str(x) + ".jpg"

        image = cv2.imread(file)
        image = cv2.resize(image, fixed_size)

        fv_hu_moments = fd_hu_moments(image)
        fv_haralick   = fd_haralick(image)
        fv_histogram  = fd_histogram(image)
        fv_lbp = fd_localBinaryPatterns(image)
        fv_zernike = fd_zernikeMoments(image)
        fv_hog = fd_histogramOrientedGradients(image)
        fv_orb = fd_orb(image)

        global_feature = np.hstack([fv_orb, fv_hog, fv_zernike, fv_lbp, fv_histogram, fv_haralick, fv_hu_moments])

        # update the list of labels and feature vectors
        labels.append(current_label)
        global_features.append(global_feature)

    print("[STATUS] processed folder: {}".format(current_label))

print("[STATUS] completed Global Feature Extraction...")

print("[STATUS] feature vector size {}".format(np.array(global_features).shape))
print("[STATUS] training Labels {}".format(np.array(labels).shape))

# encode the target labels
targetNames = np.unique(labels)
le          = LabelEncoder()
target      = le.fit_transform(labels)
print("[STATUS] training labels encoded...")

print("[STATUS] target labels: {}".format(target))
print("[STATUS] target labels shape: {}".format(target.shape))

print("[STATUS] end of training..")

glob_features = np.array(global_features)
global_labels   = np.array(target)

print("[STATUS] features shape: {}".format(glob_features.shape))
print("[STATUS] labels shape: {}".format(global_labels.shape))

print("[STATUS] training started...")

model = RandomForestClassifier(n_estimators=num_trees, random_state=seed)
name = "RandomForestClassifier"

# KFOLD test
kfold = KFold(n_splits=10, random_state=seed)
cv_results = cross_val_score(model, glob_features, global_labels, cv=kfold, scoring=scoring)
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

# 80/20 test
model.fit(glob_features, global_labels)

count = 1
results = [0,0,0,0,0,0]

for file in glob.glob(test_path + "/*.jpg"):

    image = cv2.imread(file)   
    image = cv2.resize(image, fixed_size)

    fv_hu_moments = fd_hu_moments(image)
    fv_haralick   = fd_haralick(image)
    fv_histogram  = fd_histogram(image)
    fv_lbp = fd_localBinaryPatterns(image)
    fv_zernike = fd_zernikeMoments(image)
    fv_hog = fd_histogramOrientedGradients(image) 
    fv_orb = fd_orb(image)

    test_global_feature = np.hstack([fv_orb, fv_hog, fv_zernike, fv_lbp, fv_histogram, fv_haralick, fv_hu_moments])

    # predict label of test image
    prediction = model.predict(test_global_feature.reshape(1,-1))[0]

    # Hardcoded numbers of given leaves to test
    if count < 14:
        if prediction == 0:
            results[0] += 1
    if 13 < count < 28:
        if prediction == 1:
            results[1] += 1
    if 27 < count < 41:
        if prediction == 2:
            results[2] += 1
    if 40 < count < 58:
        if prediction == 3:
            results[3] += 1
    if 57 < count < 73:
        if prediction == 4:
            results[4] += 1
    if count > 72:
        if prediction == 5:
            results[5] += 1

    # show predicted label on image
    cv2.putText(image, train_labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)
    cv2.imwrite(output_path + '/' + str(count) + '.jpg', image)
    count = count + 1

average = []
for i in range(6):
    average.append(results[i]/images_for_test[i])

print(*results, sep = ", ")
print(*average, sep = ", ")
print(sum(results)/79)