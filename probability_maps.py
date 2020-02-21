# -*- coding: utf-8 -*-
"""
TO BE IMPLEMENTED 
Gérer les effets de bords dans le calcul des HSV
"""
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature, exposure
from skimage.color import rgb2hsv
import skimage
import pandas as pd
import numpy as np

from time import time

from sklearn.base import BaseEstimator

class ImageEncoder(BaseEstimator):
    
    def __init__(self, patch_size_hsv=32, patch_size_hog=32):
        # Feature vector for each type 
        self.feature_vector = None
        self.hsv = None
        self.hog = None
        self.spatial = None
        
        self.patch_hsv = None
        
        # Size of patches for computation of features
        self.patch_size_hsv = patch_size_hsv
        self.patch_size_hog = patch_size_hog
    
    def fit(self, X):
        self.image = X
        return self.image
    
    def transform(self, X):
        return X
    
    def fit_transform(self, X, nbins_hsv=16):
        """Transform the image by computing various features"""
        # Compute spatial features
        self._compute_spatial(X)
        
        # Compute HSV features
        offset = self.patch_size_hsv // 2
        X = cv2.cvtColor(X, cv2.COLOR_BGR2HSV)
        patch_list = []
        self.hsv = None
        
        for j in range(X.shape[1]):
            for i in range(X.shape[0]):
                if (i - offset < 0) | (i + offset > X.shape[0]) | (j - offset < 0) | (j + offset > X.shape[1]):      
                    patch_hsv = np.zeros((1, 3 * nbins_hsv))
                else:
                    patch_hsv = self._compute_local_hsv(X[i - offset: i + offset,
                                                               j - offset: j + offset,:],
                                                             nbins_hsv=nbins_hsv)
                patch_list.append(patch_hsv)
                
        self.hsv = np.concatenate(tuple(patch_list), axis=0)
        
        return self.hsv
    
    def _compute_spatial(self, X):
        M = X.shape[0]
        N = X.shape[1]
        spatial_x = np.zeros((M, N))
        spatial_y = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                spatial_x[i, j] = i / M
                spatial_y[i, j] = j / N
        spatial_x = spatial_x.reshape((-1,1))
        spatial_y = spatial_y.reshape((-1,1))
        self.spatial = np.concatenate((spatial_x, spatial_y), axis=1)
        
        return self.spatial
            
            
    def _compute_local_hsv(self, X, nbins_hsv=16):
        """Compute hsv feature for a given input.
        Allows to separate the analysis into patches.
        Parameters:
            X: HSV images of the original input
        """
        # Computation of HSV features
        self.patch_hsv = None
        
        # Compute the HSV features for the entire image with a specific number of bins
        temp_patch_list = []
        for i in range(X.shape[-1]):
            temp = cv2.calcHist(images = [X[:,:,i]], channels = [0],
                                mask = None, histSize = [nbins_hsv], ranges = [0,256])
            temp_patch_list.append(temp)
        
        self.patch_hsv = np.concatenate(tuple(temp_patch_list), axis=1).reshape((-1, 1)).T
        return self.patch_hsv
  

        
# Wants to implement:
# HoG features
# HSV features
# Localized features
# Others ?
# Return df (witdh x length) x nb_features for the image inputed
        
#%%
#Set the different paths for the dahsta 
PATH = os.getcwd()
DATA_PATH = "data/FASSEG-frontal01/Train_RGB/"
ABSOLUTE_PATH = PATH + DATA_PATH

#Image
rgb_img = plt.imread(DATA_PATH + '5.bmp')

encoder = ImageEncoder()

timer = time()
features = encoder.fit_transform(rgb_img)
print(time() - timer)


#%%
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
img = cv.imread(DATA_PATH + '5.bmp')
hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
hist = cv.calcHist( [hsv], [0], None, [16], [0, 256] )
plt.imshow(hist,interpolation = 'nearest')
plt.show()

#%%
# HoG features extractor
(H, hogImage) = feature.hog(rgb_img, orientations=9, pixels_per_cell=(8, 8),
 	cells_per_block=(2, 2),	feature_vector=False, visualize=True)
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
plt.imshow(hogImage)