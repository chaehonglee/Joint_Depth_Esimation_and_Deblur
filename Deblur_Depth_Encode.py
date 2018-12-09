import os
import glob
import time
import warnings

try:
    import cPickle as pickle
except:
    import pickle

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread, imsave

from project.smoothing import bfilt
from project.knn_matting import knn_matte
from project.compute_K import getHomography
from project.update_W import update_W
from project.update_x import update_x
from project.update_D import update_D

warnings.filterwarnings('ignore')

filename = 'juul'
input_path = os.path.join(os.getcwd(), 'input', filename + '.png')

# -----------------------------------------------------------------#


blur_img = np.float32(imread(input_path)) / 255.

h, w, c = blur_img.shape

# cache for knn matting layers
with open(os.path.join(os.getcwd(), 'layers.pkl'), 'rb') as f:
    try:
        d = pickle.load(f)
    except EOFError:
        d = []

if filename in d:
    print('Layers loaded from cache.')
    layers = d[filename]
else:
    # Split the image into disjoint regions
    # 1. Apply smoothing filter
    # 2. Use Knn matting to segment the image
    bfilt_img = bfilt(blur_img, 9, 8, 0.25)

    mask_path = os.path.join(os.getcwd(), 'input', filename + '_*.png')
    mask_list = glob.glob(mask_path)
    nLayers = len(mask_list)
    layers = np.zeros((nLayers, h, w))

    for i in range(nLayers):
        mask = np.float32(imread(mask_list[i])) / 255.
        matted = knn_matte(bfilt_img, mask[:, :, 0])
        matted[matted > 0.85] = 1
        layers[i, :, :] = matted

    with open(os.path.join(os.getcwd(), 'layers.pkl'), 'wb') as f:
        pickle.dump({
            filename: layers
        }, f, protocol=2)

# initialize depth map D
depth = np.ones((h, w))

# sample camera pose and possible depths
# FIXME: combination of rot + trans
N = len(layers)
t = 10  # num sample camera poses, exposure time
s = 8  # num sample depths
V = np.linspace(5, 1, s)  # detph samples
theta = np.linspace(0.0001, 0, t)
trans = np.linspace(0.003, 0, t)

# compute Homographies : size(t)
Hom = np.zeros([s, t, 3, 3])
for i in range(s):
    fl = 1  # FIXME: find true focal length
    Hom[i] = getHomography(blur_img, fl, theta, trans, t, V[i])

# initialize x and w
x = blur_img[:, :, :]
W = np.ones((N, t)) / t

for _ in range(10):
    for _ in range(10):
        # update W
        W = update_W(W, Hom, x, layers, V, 0.5)
        # update x_i for all i's
        x = update_x(W, Hom, x, layers, V, 0.1)

    # update depth map D
    depth = update_D(blur_img, x, W, Hom, layers, V, depth, 1)

output_path = os.path.join(os.getcwd(), 'output', filename + '.png')
imsave(output_path, x)
