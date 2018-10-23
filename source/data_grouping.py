import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.misc
from scipy import signal
from scipy import misc
from scipy.signal import convolve2d
from skimage import data
from skimage.morphology import disk
from skimage.util import img_as_ubyte
import h5py
import cv2
import os 

def std_convoluted(image, N):
    im = np.array(image, dtype=float)
    im2 = im ** 2
    ones = np.ones(im.shape)

    kernel = np.ones((2 * N + 1, 2 * N + 1))
    s = convolve2d(im, kernel, mode="same")
    s2 = convolve2d(im2, kernel, mode="same")
    ns = convolve2d(ones, kernel, mode="same")
    temp = np.absolute((s2 - s ** 2 / 25) / 25)
    return np.sqrt(temp)


def normalize(x):
    mean_ker = np.ones((5, 5)) / 25
    mean = signal.convolve2d(x, mean_ker, boundary='symm', mode='same')
    std = std_convoluted(x, 2)
    blurr_image = (x - mean) / (std + 1e-5)
    return blurr_image
