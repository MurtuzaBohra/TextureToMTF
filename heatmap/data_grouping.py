#import Image
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


def grouping(path_group_indices, path_patches, path_roi_images,patch_size):
    bdry_mrg = patch_size/2
    groups = open(path_group_indices, 'r')  # indices for each group
    for i, f in enumerate(groups):
        print ('The value of f: ', f)
        lower, upper = f.split(',')
        lower = int(lower)
        upper = int(upper)
        size = range(lower, upper + 1)
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
        total_nump = 0
        nump = {}
        content = {k: [] for k in size}
        for gr_i, gr_v in enumerate(size):
            with open(path_patches+str(gr_v)+'.txt') as f:
    	        content[gr_v] = f.readlines()
    	        total_nump = total_nump + len(content[gr_v])
    	        nump[gr_v] = len(content[gr_v])
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------

        X = np.zeros((total_nump, patch_size, patch_size))
    #    Y = np.zeros((total_nump))

        for gr_i, gr_v in enumerate(size):
            filename = path_roi_images + str(gr_v) +'.png'
            x = plt.imread(filename)

            r, c = x.shape
            blurr_image = normalize(x)  # Normalize using 5 X 5 % kernel


            content[gr_v] = [x.strip() for x in content[gr_v]]
            idx = 1
            ind_sum = 0
            for g in range(0,gr_i):
                ind_sum = ind_sum + nump[size[g]]

            while (idx<=len(content[gr_v])):
                lst = content[gr_v][idx-1].split(',')
                midr=int(lst[0])
                midc=int(lst[1])

                r1 = max(0, midr-bdry_mrg)
                r2 = min(midr+bdry_mrg-1, r-1)
                c1 = max(midc-bdry_mrg, 0)
                c2 = min(midc+bdry_mrg-1, c-1)

                croppedimg = blurr_image[r1:r2+1,c1:c2+1]
                a,b = croppedimg.shape
                if(a!=48 or b!=48):
                    croppedimg = cv2.resize(croppedimg, (48,48))
                    print('patch is resized to 48X48')

                X[ind_sum + idx-1, :, :] = croppedimg
                idx = idx + 1
        #with h5py.File(path_group+'group_nice_fs%d.hdf5' % (i+200), 'w') as hf:
        #    hf.create_dataset('X', data=X)
    return(X)




'''def grouping(path_group_indices, path_patches, path_roi_images,patch_size):
    bdry_mrg = patch_size/2
    groups = open(path_group_indices, 'r')  # indices for each group
    for i, f in enumerate(groups):
        print ('The value of f: ', f)
        lower, upper = f.split(',')
        lower = int(lower)
        upper = int(upper)
        size = range(lower, upper + 1)
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------
        total_nump = 0
        nump = {}
        content = {k: [] for k in size}
        for gr_i, gr_v in enumerate(size):
            with open(path_patches+str(gr_v)+'.txt') as f:
                content[gr_v] = f.readlines()
                total_nump = total_nump + len(content[gr_v])
                nump[gr_v] = len(content[gr_v])
    #----------------------------------------------------------------------------------
    #----------------------------------------------------------------------------------

        X = np.zeros((11, patch_size, patch_size))
    #    Y = np.zeros((total_nump))

        idx = 1

        for gr_i, gr_v in enumerate(size):
            filename = path_roi_images + str(gr_v) +'.png'
            x = plt.imread(filename)

            r, c = x.shape
            blurr_image = normalize(x)  # Normalize using 5 X 5 % kernel


            content[gr_v] = [x.strip() for x in content[gr_v]]
            ind_sum = 0
            #####
            #for g in range(0,gr_i):
                #ind_sum = ind_sum + nump[size[g]]

                #while (idx<=len(content[gr_v])):
                #lst = content[gr_v][idx-1].split(',')
            ####
            midr = 24
            midc = 24

            r1 = max(1, midr-bdry_mrg)
            r2 = min(midr+bdry_mrg, r-1)
            c1 = max(midc-bdry_mrg, 0)
            c2 = min(midc+bdry_mrg, c-1)

            croppedimg = blurr_image[r1:r2+1,c1:c2+1]
            a,b = croppedimg.shape
            if(a!=48 or b!=48):
                croppedimg = cv2.resize(croppedimg, (48,48))

            X[ind_sum + idx-1, :, :] = croppedimg
            idx = idx + 1
        #with h5py.File(path_group+'group_nice_fs%d.hdf5' % (i+200), 'w') as hf:
        #    hf.create_dataset('X', data=X)
    return(X)
'''