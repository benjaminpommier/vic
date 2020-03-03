# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 16:31:41 2020

@author: Benjamin Pommier
"""

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import glob
import pandas as pd
import pickle
import argparse

#Loading data path
# PATH = 'features/features_*[0-9].csv'

def main(args):
    """
    Compute probability maps for a given input using a pretrained model.
    Probability maps are save accordingly in a subfolder probability_maps_test/.

    Parameters
    ----------
    path : str
        Path to the folder containing the data. e.g.: 'data/dataset1/'

    Returns
    -------
    None.

    """
    PATH = args.path
    path_list = sorted(glob.glob(PATH))
    
    #Loading model
    model = pickle.load(open('model/'+args.model+'.sav', 'rb'))
    
    for pth in path_list:
        X = pd.read_csv(pth)
        idx = X.image.max()
        print('------- ' + str(idx) + ' -------')
        pred = pd.DataFrame(model.predict_proba(X))
        filename = 'probability_maps_perso/%.d.csv'%(idx)   
        pred.to_csv(open(filename, 'w'), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path",  type=str)
    parser.add_argument("--model", type=str, default='random_forest')
    args = parser.parse_args()
    main(args)
    
