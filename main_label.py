# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:22:46 2020

@author: Benjamin Pommier
"""

from features_encoder import ImageEncoder
from label_encoder import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import glob

### NOT USED IN THE PIPELINE

# Paths
ORIGINAL_PATH = 'data/FASSEG-frontal03/Original/'
LABELED_PATH = 'data/FASSEG-frontal03/Labeled/'
EXTENSION = 'jpg'

orignal_data = glob.glob(ORIGINAL_PATH + '*.{}'.format(EXTENSION))
labeled_data = glob.glob(LABELED_PATH + '*.{}'.format(EXTENSION))

label_encoder = LabelEncoder()
image_encoder = ImageEncoder(patch_size_hsv=16, patch_size_hog=16, nbins_hsv=16)

#Loading & encoding labels
encoding = []
for i, lbl in enumerate(labeled_data):
    im = plt.imread(lbl)
    result = pd.DataFrame(label_encoder.fit_transform(im))
    result['image'] = i
    encoding.append(result)
    
encoding = pd.concat(encoding, axis=0)