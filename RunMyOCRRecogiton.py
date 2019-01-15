import numpy as np
import train
import test
from train import get_threshold, extract_features, normalize, store_features
from test import collect 
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops, moments, moments_central,moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import sys

def training():
    files = ['a', 'd', 'm', 'n', 'o', 'p', 'q', 'r', 'u', 'w']
    for f in files:
        train.training_test(f, 0)

def recognition_test(path, show):
    (data_feature_arr, data_label, mean, std) = store_features(0)

    (boxes, test_feature_arr) = collect(path, mean, std)
    sys.stdout.write('Number of components: ')
    print len(boxes)

    D = cdist(test_feature_arr, data_feature_arr)
    D_index = np.argsort(D, axis=1)

    pkl_file = open(path+'_gt.pkl', 'rb')
    mydict = pickle.load(pkl_file)
    pkl_file.close()
    classes = mydict['classes']
    locations = mydict['locations']
	
    class_list = classes.tolist()
    location_list = locations.tolist()

    num = len(class_list)
    error = 0
    Ytrue = []
    Ypred = []

    if show == 1:
        io.imshow(D)
        plt.title('Distance Matrix')
        io.show()

        img = io.imread('./images/' + path + '.bmp' )
        hist = exposure.histogram(img)
        th = get_threshold('./images/' + path + '.bmp')
        img_binary = (img < th).astype(np.double)
        img_label = label(img_binary, background=0)
        regions = regionprops(img_label)
        io.imshow(img_binary)
        ax = plt.gca()

    for i in range(num):
        for j in range(len(boxes)):
            minr = boxes[j][0]
            maxr = boxes[j][1]
            minc = boxes[j][2]
            maxc = boxes[j][3]
            if location_list[i][1] < minr or location_list[i][1] > maxr or location_list[i][0] < minc or location_list[i][0] > maxc:
                continue
            index = D_index[j][0]
            Ypred.append(chr(data_label[index]))
            Ytrue.append(class_list[i])

            if show == 1:
                ax.add_patch(Rectangle((minc, minr), maxc - minc, maxr - minr, fill=False, edgecolor='red', linewidth=1))
                ax.text(maxc, maxr, chr(data_label[index]), color='green')

            if class_list[i] != chr(data_label[index]):
                error += 1
    if show == 1:
        plt.title('Bounding Boxes')
        io.show()
        confM = confusion_matrix(Ytrue, Ypred)
        io.imshow(confM)
        plt.title('Confusion Matrix')
        io.show()
    sys.stdout.write('Recognition Test Accuracy: ')
    print 1- error/float(num)
    
recognition_test("test1",1)
