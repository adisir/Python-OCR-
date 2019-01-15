import numpy as np
from skimage.measure import label, regionprops, moments, moments_central,moments_normalized, moments_hu
from skimage import io, exposure
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
from train import get_threshold,extract_features, normalize, store_features
import sys


def collect(path, mean, std):
    img = io.imread('./images/' + path + '.bmp' )
    hist = exposure.histogram(img)
    th = get_threshold('./images/' + path + '.bmp')
    img_binary = (img < th).astype(np.double)
    img_label = label(img_binary, background=0)
    regions = regionprops(img_label)
    boxes = []
    features = []
    for props in regions:
        box = []
        minr, minc, maxr, maxc = props.bbox
        if maxc - minc < 10 or maxr - minr < 10 or maxc - minc > 120 or maxr - minr > 120:
            continue
        box.append(minr)
        box.append(maxr)
        box.append(minc)
        box.append(maxc)
        boxes.append(box)

        roi = img_binary[minr:maxr, minc:maxc]
        m = moments(roi)
        cr = m[0, 1] / m[0, 0]
        cc = m[1, 0] / m[0, 0]
        mu = moments_central(roi, cr, cc)
        nu = moments_normalized(mu)
        hu = moments_hu(nu)
        features.append(hu)

    feature_arr = normalize(features, mean, std)
    return (boxes, feature_arr)



