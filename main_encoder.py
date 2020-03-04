# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 20:22:46 2020

@author: Benjamin Pommier
"""

from features_encoder import ImageEncoder, compute_features, save
from label_encoder import LabelEncoder, encode_labels
import matplotlib.pyplot as plt
import pandas as pd
import glob
import time

#%%
# Define the paths
ORIGINAL_PATH = 'data/FASSEG-frontal03/Original'
LABELED_PATH = 'data/FASSEG-frontal03/Labeled/'
EXTENSION = 'jpg'

# Extract all the image paths
original_data = sorted(glob.glob(ORIGINAL_PATH + '*.{}'.format(EXTENSION)))
labeled_data = glob.glob(LABELED_PATH + '*.{}'.format(EXTENSION))

# Encode labels & images with the predefined Encoders
label_encoder = LabelEncoder()
image_encoder = ImageEncoder(patch_size_hsv=16, patch_size_hog=16, nbins_hsv=16)

labels = encode_labels(labeled_data)
save(labels, 'labels')

features = compute_features(original_data, encoder=image_encoder, batch=1)
