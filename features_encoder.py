# -*- coding: utf-8 -*-
"""
@author: Benjamin Pommier
"""

# Importing useful package
import cv2
from skimage import feature
import numpy as np

from time import time

from sklearn.base import BaseEstimator

class ImageEncoder(BaseEstimator):
    '''Create a class to encode the image and return all the features
    Parameters:
        patch_size_hsv (int) : size of the patch used to compute the HSV features (e.g.: 16, 32, 64)
        patch_size_hog (int) : size of the patch used to compute the HOG features (e.g.: 16, 32, 64)
        nbins_hsv (int) : number of bins used in the computation of hsv histograms
    '''
    
    def __init__(self, patch_size_hsv=32, patch_size_hog=32, nbins_hsv=16):
        # Feature vector for each type 
        self.feature_vector = None
        self.hsv = None
        self.hog = None
        self.spatial = None
        
        # Attribute use in the computation of HSV features
        self.patch_hsv = None
        
        # Size of patches for computation of features*
        self.nbins_hsv = nbins_hsv
        self.patch_size_hsv = patch_size_hsv
        self.patch_size_hog = patch_size_hog
    
    
    def fit(self, X):
        '''Identity function, implemented only to fit in a sklearn pipeline
        Parameters:
            X (array-like) : image to be processed
        Return:
            X (array) : same image
        '''
        return X
    
    
    def transform(self, X):
        '''Cf. fit_transform for more details. Does the same thing.
        Implemented to be used in a sklearn pipeline
        '''
        return self.fit_transform(X, nbins_hsv=self.nbins_hsv)
    
    
    def fit_transform(self, X):
        """Transform the image by computing alll the features
        Parameters:
            X (array) : input image to be processed
        Returns : 
            feature_vector (array) : (#pixels, #feature) vector, transformed image
        """
        # Compute spatial features
        print('--- Computing spatial features --- ')
        self._compute_spatial(X)
        
        # Compute HSV features
        
        print('--- Computing HSV features --- ')
        # Initialisation of useful variables
        M = X.shape[0]
        N = X.shape[1]
        #Conversion & padding
        window_hsv = self.patch_size_hsv // 1
        X_hsv = np.pad(X, pad_width=((window_hsv, window_hsv), (window_hsv, window_hsv), (0,0)),
                       mode='constant', constant_values=0)
        X_hsv = cv2.cvtColor(X_hsv, cv2.COLOR_BGR2HSV)
        patch_list = []
        self.hsv = None
        
        for j in range(N):
            for i in range(M):
                patch_hsv = self._compute_local_hsv(X_hsv[i : i + window_hsv,
                                                               j : j + window_hsv,:],
                                                             nbins_hsv=self.nbins_hsv)
                patch_list.append(patch_hsv)
                
        self.hsv = np.concatenate(tuple(patch_list), axis=0)
        
        #Compute HoG features
        print('--- Computing HoG features --- ')
        self._compute_hog(X)
                
        self.feature_vector = np.concatenate((self.spatial, self.hsv, self.hog), axis=1)
        
        return self.feature_vector
    
    
    def _compute_spatial(self, X):
        """Method to compute spatial features
        Parameters:
            X (array) : input image
        Return:
            spatial (array) : (#pixels, 2) vector containing only spatial features
        """
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
            

    def _compute_local_hsv(self, X, nbins_hsv=None):
        """Compute hsv feature for a given input.
        Allows to separate the analysis into patches.
        Parameters:
            X: HSV images of the original input
            nbins_hsv (int) : (optional) number of bins used in the computation of hsv histograms 
                                if different from default parameters
        """
        if nbins_hsv is None:
            nbins_hsv = self.nbins_hsv
            
        # Computation of HSV features
        self.patch_hsv = None
        
        # Compute the HSV features for the entire image with a specific number of bins
        temp_patch_list = []
        for i in range(X.shape[-1]):
            temp = cv2.calcHist(images = [X[:,:,i]], channels = [0],
                                mask = None, histSize = [nbins_hsv], ranges = [0,256])
            temp_patch_list.append(temp)
        
        self.patch_hsv = np.concatenate(tuple(temp_patch_list), axis=1).reshape((1, -1))
        return self.patch_hsv
        
    
    def _compute_hog(self, X, window_hog=None):
        """Compute HoG feature for a given input
        Parameters:
            X (array) : Original input / image
            window_hog (int) : (optional) size of the window used to compute the features 
                                if different from default parameters
        Return:
            hog (array) : HoG features per pixels
        """
        #Initialisation
        self.hog = None
        M = X.shape[0]
        N = X.shape[1]
              
        #Handling customized patch_size
        if window_hog is None:
            window_hog = self.patch_size_hog // 1
        
        #Padding of the image
        X_hog = np.pad(X, pad_width=((window_hog, window_hog), (window_hog, window_hog), (0,0)),
                       mode='constant', constant_values=0)
        hog_features = []
        
        #Iterations over the windows to compute local hog features
        for j in range(N):
            for i in range(M):
                patch_hog = X_hog[i : i + window_hog,
                                  j : j + window_hog, :]
                hog_features.append(feature.hog(patch_hog, orientations=9, pixels_per_cell=(8,8),
                                                cells_per_block=(2,2), visualize=False, transform_sqrt=False, 
                                                feature_vector=True).reshape((1, -1)))
            
        self.hog = np.concatenate(tuple(hog_features), axis=0)
        return self.hog

        
#%% TESTING

# DATA_PATH = "data/FASSEG-frontal01/Train_RGB/"
# rgb_img = plt.imread(DATA_PATH + '5.bmp')

# encoder = ImageEncoder(patch_size_hsv=16, patch_size_hog=16, nbins_hsv=16)

# timer = time()
# features = encoder.fit_transform(rgb_img)
# print(time() - timer)