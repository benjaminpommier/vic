# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 08:44:29 2020

@author: Benjamin Pommier
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
import glob
import pandas as pd
import pickle

# #Data
exception = 13
X_train = pd.read_csv('features/features_train.csv')
max_im = X_train.image.max()
X_train = X_train[(X_train.image <= max_im) & (X_train.image != exception)]

X_dev = pd.read_csv('features/features_dev.csv')

labels = pd.read_csv('features/labels.csv')
y_train = labels[(labels.image <= max_im) & (labels.image != exception)].label

y_dev = labels[(labels.image > max_im) & (labels.image < 31)].label

#Model RF
model = GradientBoostingClassifier()
params = {'n_estimators': [10], 'min_samples_leaf': [100]}
# params = {'n_estimators': [10, 30, 50], 'min_samples_leaf': [1, 10, 100, 1000]}
gridsearch = GridSearchCV(model, param_grid=params, cv=3, verbose=3, n_jobs=-1)

#Fitting
gridsearch.fit(X_train, y_train)
best_model = gridsearch.best_estimator_
# best_model.n_jobs = -1 #Setting te parameter afterwards otherwise None
best_model.fit(X_train, y_train)

#Save the model to disk
filename_model = 'gradient_boosting.sav'
pickle.dump(best_model, open(filename_model, 'wb'))

#%%
#Prediction

best_model_load = pickle.load(open('model/random_forest.sav', 'rb'))
#Training set
y_pred_train = best_model_load.predict(X_train)
y_probas_train = best_model_load.predict_proba(X_train)
print(classification_report(y_train, y_pred_train))

#Test set
y_pred_dev= best_model_load.predict(X_dev)
y_probas_dev= best_model_load.predict_proba(X_dev)
print(classification_report(y_dev, y_pred_dev))

#%%
import matplotlib.pyplot as plt

def visualize(labels, features=None, model=None, image_num=None, predict=True):
    ORIGINAL_PATH = 'data/FASSEG-frontal03/Original/'
    LABELED_PATH = 'data/FASSEG-frontal03/Labeled/'
    EXTENSION = '.jpg'

    if image_num < 10:
        or_image = plt.imread(ORIGINAL_PATH + '00' + str(int(image_num)) + EXTENSION)
        lbl_image = plt.imread(LABELED_PATH + '00' + str(int(image_num)) + EXTENSION)
    elif (image_num >= 10) & (image_num < 100):
        or_image = plt.imread(ORIGINAL_PATH + '0' + str(int(image_num)) + EXTENSION)
        lbl_image = plt.imread(LABELED_PATH + '0' + str(int(image_num)) + EXTENSION)
    else:
        raise NotImplementedError
    
    # Prediction for a given image
    if predict:
        X = features[features.image == image_num]
        pred = model.predict_proba(X)
    else:
        pred = labels
    
    try:
        #Display
        plt.figure(figsize=(20,20))
        plt.subplot(331)
        plt.title('Original')
        plt.imshow(or_image)
        
        plt.subplot(332)
        plt.title('Ground truth')
        plt.imshow(lbl_image)
        
        plt.subplot(334)
        plt.title('Mouth') #nose eyes hair background skin
        plt.imshow(pred[:, 0].reshape((512, -1)))
    
        plt.subplot(335)
        plt.title('Nose') #nose eyes hair background skin
        plt.imshow(pred[:, 1].reshape((512, -1)))
        
        plt.subplot(336)
        plt.title('Eyes') #nose eyes hair background skin
        plt.imshow(pred[:, 2].reshape((512, -1)))
        
        plt.subplot(337)
        plt.title('Hair') #nose eyes hair background skin
        plt.imshow(pred[:, 3].reshape((512, -1)))
        
        plt.subplot(338)
        plt.title('Background') #nose eyes hair background skin
        plt.imshow(pred[:, 4].reshape((512, -1)))
        
        plt.subplot(339)
        plt.title('Skin') #nose eyes hair background skin
        plt.imshow(pred[:, 5].reshape((512, -1)))
        
        plt.tight_layout()
        plt.show()
    except:
        pass
    
# visualize(model=best_model, features=X_dev, labels=y_dev, image_num=28)
im_num = 37
file = pd.read_csv('probability_maps_test/' + str(im_num) + '.csv').to_numpy()
visualize(labels = file, image_num=im_num, predict=False)