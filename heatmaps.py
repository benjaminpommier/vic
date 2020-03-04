# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 08:52:26 2020

@author: Benjamin Pommier
"""

import pickle
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

classif_train = pickle.load(open('model/results/all_logreg_results_train.pkl', 'rb'))
classif_test = pickle.load(open('model/results/all_logreg_results_dev.pkl', 'rb'))
df_train = pd.DataFrame(classif_train)
df_test = pd.DataFrame(classif_test)

naming = {'0': 'mouth', '1': 'nose', '2': 'eyes', '3': 'hair', '4': 'background', '5': 'skin'}

df_train = df_train.rename(columns=naming).drop(index=['support'])
df_test = df_test.rename(columns=naming).drop(index=['support'])

sns.heatmap(df_train, cbar=False, annot=True, fmt='.2f')
plt.show()
sns.heatmap(df_test, cbar=False, annot=True, fmt='.2f')
plt.show()