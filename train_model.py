# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 08:44:29 2020

@author: Benjamin Pommier
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import glob
import pandas as pd
import pickle
import numpy as np

# Load data
def load_data(path_train=None, path_dev=None, path_labels=None, partial='all'):
    
    if path_train is None:
        path_train = 'features/features_train.csv'
    if path_dev is None:
        path_dev = 'features/features_dev.csv'
    if path_labels is None:
        path_labels = 'features/labels.csv'
    
    hog = list(np.arange(50, 86, 1))
    hsv = list(np.arange(2, 50, 1))
    spatial = list([0,1])
    image = list([86])
    
    if partial == 'all':
        usecol = spatial + hsv + hog + image
    elif partial == 'spatial_hsv':
        usecol = spatial + hsv + image
    elif partial == 'spatial_hog':
        usecol = spatial + hog + image
    elif partial == 'hsv_hog':
        usecol = hsv + hog + image
    elif partial == 'hsv':
        usecol = hsv + image
    elif partial == 'hog':
        usecol = hog + image
    
    exception = 13 #Problematic image, label shape differs from image shape
    
    #Loading training set
    print('--- Loading Training Set ---')
    X_train = pd.read_csv(path_train, usecols=usecol)
    max_im = X_train.image.max()
    X_train = X_train[(X_train.image <= max_im) & (X_train.image != exception)]
    
    #Loading dev set
    print('--- Loading Dev Set ---')
    X_dev = pd.read_csv(path_dev, usecols=usecol)
    
    print('--- Loading Ground Truth ---')
    labels = pd.read_csv(path_labels)
    y_train = labels[(labels.image <= max_im) & (labels.image != exception)].label
    
    y_dev = labels[(labels.image > max_im) & (labels.image < 31)].label
    
    print('--- END OF DATA LOADING ---')
    return X_train, X_dev, y_train, y_dev

def train(X_train, y_train, model=None, gridsearch=False, filename_model='None'):
    if model is None:
        model = RandomForestClassifier(n_jobs=-1)
        #Model RF
    
    if gridsearch:
        if type(model) == type(LogisticRegression()):
            params = {'C':[0.01, 0.1, 1, 10]}
        elif type(model) == type(RandomForestClassifier()):
            params = {'n_estimators': [10, 30, 50], 'min_samples_leaf': [10, 100]}
        elif type(model) == type(SVC()):
            params = {'C': [0.1, 1, 10]}
    
        gridsearch = GridSearchCV(model, param_grid=params, cv=3, verbose=2, n_jobs=-1)
    
        gridsearch.fit(X_train, y_train)
        model = gridsearch.best_estimator_
        print(model)
        model.n_jobs = -1 #Setting te parameter afterwards otherwise None
        model.fit(X_train, y_train)
    else:
        model.n_jobs = -1
        model.fit(X_train, y_train)
    
    #Save the model to disk
    pickle.dump(model, open(filename_model+'.sav', 'wb'))

def evaluate(X_train, X_dev, y_train, y_dev, type_model='rf'):
    if type_model == 'rf' :
        best_model = pickle.load(open('model/random_forest.sav', 'rb')) 
    elif type_model == 'logreg':
        best_model = pickle.load(open('model/logreg.sav', 'rb'))
    elif type_model == 'svc':
        best_model = pickle.load(open('model/svc.sav', 'rb'))
    else:
        best_model = pickle.load(open(type_model+'.sav', 'rb'))
    
    #Training set
    y_pred_train = best_model.predict(X_train)
    y_probas_train = best_model.predict_proba(X_train)
    print(classification_report(y_train, y_pred_train))
    pickle.dump(classification_report(y_train, y_pred_train, output_dict=True),
                open(type_model+'_results_train.pkl', 'wb'))
    
    #Dev set
    y_pred_dev= best_model.predict(X_dev)
    y_probas_dev= best_model.predict_proba(X_dev)
    print(classification_report(y_dev, y_pred_dev))
    pickle.dump(classification_report(y_dev, y_pred_dev, output_dict=True),
                open(type_model+'_results_dev.pkl', 'wb'))
    
    return y_probas_train, y_probas_dev

#%% Data Loading
partial = ['all']
for prt in partial:
    X_train, X_dev, y_train, y_dev = load_data(partial=prt)
    
    rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=100, n_jobs=-1)
    logreg = LogisticRegression(C=1, n_jobs=-1)
    # svc = SVC()
    
    train(X_train, y_train, model=rf, gridsearch=True, filename_model=prt+'_random_forest')
    train(X_train, y_train, model=logreg, gridsearch=True, filename_model=prt+'_logreg')
    
    evaluate(X_train, X_dev, y_train, y_dev, type_model=prt+'_random_forest')
    evaluate(X_train, X_dev, y_train, y_dev, type_model=prt+'_logreg')


#%%Visualisation
    
def visualize(labels=None, features=None, model=None, image_num=None, predict=True):
    ORIGINAL_PATH = 'data/FASSEG-frontal03/Original/'
    # ORIGINAL_PATH = 'data/perso/'
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
        plt.imshow(pred[:, 0].reshape((512, -1)), cmap='gray')
    
        plt.subplot(335)
        plt.title('Nose') #nose eyes hair background skin
        plt.imshow(pred[:, 1].reshape((512, -1)), cmap='gray')
        
        plt.subplot(336)
        plt.title('Eyes') #nose eyes hair background skin
        plt.imshow(pred[:, 2].reshape((512, -1)), cmap='gray')
        
        plt.subplot(337)
        plt.title('Hair') #nose eyes hair background skin
        plt.imshow(pred[:, 3].reshape((512, -1)), cmap='gray')
        
        plt.subplot(338)
        plt.title('Background') #nose eyes hair background skin
        plt.imshow(pred[:, 4].reshape((512, -1)), cmap='gray')
        
        plt.subplot(339)
        plt.title('Skin') #nose eyes hair background skin
        plt.imshow(pred[:, 5].reshape((512, -1)), cmap='gray')
        
        plt.tight_layout()
        plt.show()
    except:
        pass
    

# im_num = 5
# file = pd.read_csv('probability_maps_perso/' + str(im_num) + '.csv').to_numpy()
# visualize(labels = file, image_num=im_num, predict=False)
        
im_num = 9
mdl = pickle.load(open('model/random_forest.sav', 'rb')) 
visualize(features=X_train, model=mdl, image_num=im_num, predict=True)