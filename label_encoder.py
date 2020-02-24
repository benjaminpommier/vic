# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:22:46 2020

@author: Benjamin Pommier
"""

from features_encoder import ImageEncoder
import matplotlib.pyplot as plt
import numpy as np
import glob

from sklearn.base import BaseEstimator

class LabelEncoder(BaseEstimator):
    
    def __init__(self, img=None):
        self.img = img
        self.dict_colors = {'mouth': np.array([0,255,0]), 'nose': np.array([0,255,255]), 'eyes': np.array([0,0,255]),
                            'hair': np.array([130,0,0]), 'background': np.array([255,0,0]), 'skin': np.array([255,255,0])}
        self.correspondance = {'mouth': 0, 'nose': 1, 'eyes': 2, 'hair': 3, 'background': 4, 'skin': 5}
        self.labels = None
        self.labels_mapped = None
        
    def fit(self, img):
        return img
    
    def transform(self, img):
        return self.fit_transform(img)
    
    def fit_transform(self, image):
        M = image.shape[0]
        N = image.shape[1]
        self.img = np.copy(image)
        self.labels = []
        values = list(self.dict_colors.values())
        keys = list(self.dict_colors.keys())
        
        for j in range(N):
            for i in range(M):
                test = values - image[i][j]
                idx = np.argmin([np.linalg.norm(x) for x in test])
                self.img[i][j] = values[idx]
                self.labels.append(keys[idx])
        
        self.labels_mapped = np.array(list(map((lambda x: self.correspondance[x]), self.labels)))
        
        return self.labels_mapped


#%% TESTING
        
# encoder = LabelEncoder()

# image = plt.imread(labeled_data[3])
# labels = encoder.fit_transform(image)
# cleaned_image = encoder.img

# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122) 
# plt.imshow(cleaned_image)
# plt.show()
