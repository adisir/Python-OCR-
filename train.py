import numpy as np
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central,moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import sys
import cv2


def get_threshold(image_path):
    img = cv2.imread(image_path,0)
    blur = cv2.GaussianBlur(img,(5,5),0)

    img_binary = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)  
    kernel = np.ones((3,3),np.uint8)
    closing = cv2.morphologyEx(~img_binary, cv2.MORPH_CLOSE, kernel)

    
    return closing

def extract_features(path, show, tag):
    img = io.imread('./images/' + path + '.bmp')
    hist = exposure.histogram(img)
    th = get_threshold('./images/' + path + '.bmp')
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background=0)

    # Show images
    if show == 1:
        io.imshow(img)
        plt.title('Original Image')
        io.show()

        plt.bar(hist[1], hist[0])
        plt.title('Histogram')
        plt.show()

        io.imshow(img_binary)
        plt.title('Binary Image')
        io.show()

        io.imshow(img_label)
        plt.title('Labeled Image')
        io.show()

    regions = regionprops(img_label)
    if show == 1:
        io.imshow(img_binary)
        ax = plt.gca()

    features = []

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc < 10 or maxr - minr < 10 or maxc - minc > 120 or maxr - minr > 120:
            continue
        if show == 1:
            ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        features.append(hu)
        if(len(path)==1):
            tag.append(ord(path))

    if show == 1:
        plt.title('Bounding Boxes')
        io.show()
    return features


def store_features(show):
    files = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
    feature_arr = []
    label = []

    for i in files:
        feature_arr += extract_features(i, show, label)

    mean = np.mean(feature_arr)
    std = np.std(feature_arr)
    feature_arr = normalize(feature_arr, mean, std)
    return (feature_arr, label, mean, std)


def normalize(features, mean, std):
    for i in range(len(features)):
        features[i] = (features[i] - mean) / float(std)
    return features

def training_test(path, show):
    (data_feature_arr, data_label, mean, std) = store_features(show)
    temp = []
    test_features = extract_features(path, 0, temp)
    test_feature_arr = normalize(test_features, mean, std)
    D = cdist(test_feature_arr, data_feature_arr)
    D_index = np.argsort(D, axis=1)
    error = 0

    img = io.imread('./images/' + path + '.bmp' )
    hist = exposure.histogram(img)
    th = get_threshold('./images/' + path + '.bmp')
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background = 0)
    regions = regionprops(img_label)
    io.imshow(img_binary)
    ax = plt.gca()
    i = 0

    for props in regions:
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc < 10 or maxr - minr < 10 or maxc - minc > 120 or maxr - minr > 120:
            continue
        index = D_index[i][1]
        ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
        ax.text(maxc, maxr, chr(data_label[index]), color='green')
        if path != chr(data_label[index]):
            error += 1
        i += 1
    plt.title('Bounding Boxes')
    io.show()
    sys.stdout.write('Training test rate for ' + path + '.bmp')
    print 1 - error/float(len(test_feature_arr))




