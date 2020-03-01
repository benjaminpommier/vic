# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:22:46 2020

@author: Benjamin Pommier
"""

import cv2
from skimage import feature
import numpy as np
import re
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time

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
        
        for i in range(M):
            for j in range(N):
                test = values - image[i][j]
                idx = np.argmin([np.linalg.norm(x) for x in test])
                self.img[i][j] = values[idx]
                self.labels.append(keys[idx])
        
        self.labels_mapped = np.array(list(map((lambda x: self.correspondance[x]), self.labels)))
        
        return self.labels_mapped


#Loading & encoding labels
def encode_labels(labeled_data_path, encoder=None, num=None):
    if encoder is None:
        encoder = LabelEncoder()
        
    if num is None:
        num = len(labeled_data_path)
    
    print('--- Encoding Labels ---')
    encoding = []
    timer = time.time()
    for lbl in labeled_data_path[:num]:
        idx = re.findall(r'\d+', lbl)[-1]
        print('Image ' + idx + ' -- %.3f'%(time.time() - timer))
        im = plt.imread(lbl)
        result = pd.DataFrame(encoder.fit_transform(im))
        result['image'] = int(idx)
        encoding.append(result)
        
    encoding = pd.concat(encoding, axis=0).rename(columns={0: 'label'})
    return encoding

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
