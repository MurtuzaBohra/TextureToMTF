from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D, Lambda, BatchNormalization
import pool
import h5py
import numpy as np
from scipy import stats
from keras.layers import LeakyReLU
from keras import backend as K
import cv2
import os 
import matplotlib.pyplot as plt
K.set_image_dim_ordering('th')

batch_size = 100
patch_size = 48
bdry_mrg = patch_size/2
patch_dir = './patches/'
MTF_file = './MTF_scores.txt'
img_dir = './images/'
weight_file = 'weight.hdf5'
img_format = '.jpg'
total_images = 10

model = Sequential()
model.add(Conv2D(40, 5, 5, input_shape=(1,patch_size, patch_size)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(BatchNormalization())

model.add(Conv2D(80, 5, 5))
model.add(Lambda(pool.min_max_pool2d, output_shape=pool.min_max_pool2d_output_shape))
model.add(BatchNormalization())

model.add(Flatten())

model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(BatchNormalization())
model.add(LeakyReLU(0.01))

model.add(Dropout(0.2))
model.add(Dense(1024))
model.add(LeakyReLU(0.01))

model.add(Dense(1))
model.load_weights(weight_file)
model.compile(loss='mse', optimizer = 'adam')
print (model.summary())


#----------------------------------------------------------------------------------------------------------------------


import data_grouping

prediction = []

for i in range(total_images):
    patch_file = patch_dir+str(i+1)+'.txt'
    img_filename = img_dir+str(i+1)+img_format
    with open(patch_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]

    x = plt.imread(img_filename)
    r, c = x.shape
    blurr_image = AF_data_grouping.normalize(x)

    num_patches = len(content)

    X_test = np.zeros((num_patches, patch_size, patch_size))
    for patch in range(num_patches):

        lst = content[patch].split(',')
        midr=int(lst[0])
        midc=int(lst[1])

        r1 = int(midr-bdry_mrg)
        r2 = int(midr+bdry_mrg-1)
        c1 = int(midc-bdry_mrg)
        c2 = int(midc+bdry_mrg-1)

        croppedimg = blurr_image[r1:r2+1,c1:c2+1]
        a,b = croppedimg.shape
        if(a!=patch_size or b!=patch_size):
            croppedimg = cv2.resize(croppedimg, (patch_size,patch_size))
            print('patch is resized to 48X48')

        X_test[patch, :, :] = croppedimg

    X_test = np.expand_dims(X_test, axis = 1)
    prediction.extend(np.mean(model.predict(X_test, verbose=1)))

#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------
print(len(prediction))
thefile = open(MTF_file, 'w')
for item in prediction:
    thefile.write("%s\n" % item[0])