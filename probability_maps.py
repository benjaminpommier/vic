# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""
import cv2
import os
import matplotlib.pyplot as plt
from skimage import feature, exposure
from skimage.color import rgb2hsv

#Set the different paths for the data 
PATH = os.getcwd()
DATA_PATH = "data/FASSEG-frontal01/Train_RGB/"
ABSOLUTE_PATH = PATH + DATA_PATH

#Image
rgb_img = plt.imread(DATA_PATH + '5.bmp')

# HoG features extractor
(H, hogImage) = feature.hog(rgb_img, orientations=9, pixels_per_cell=(8, 8),
	cells_per_block=(2, 2),	feature_vector=False, visualize=True)
hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
hogImage = hogImage.astype("uint8")
plt.imshow(hogImage)

#HSV
hsv_img = rgb2hsv(rgb_img)
hue_img = hsv_img[:, :, 0]
sat_img = hsv_img[:, :, 1]
value_img = hsv_img[:, :, 2]

