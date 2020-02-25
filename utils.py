import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.ndimage import filters



### IMPORT FUNCTIONS

def import_dataset(path_images,sub_dir_original,sub_dir_labeled,dataset_size):
    
    # Importing dataset
    
    names = sorted([img for img in listdir(path_images+sub_dir_original) if isfile(join(path_images+sub_dir_original, img))], reverse=True)
    names = names[1:dataset_size+1]

    images = [plt.imread(path_images+sub_dir_original+name) for name in names]
    raw_labels = [plt.imread(path_images+sub_dir_labeled+name) for name in names]
    
    # Removing image from dataset if dimension is not correct

    to_remove = []
    for i in range(len(images)):
        if (images[i].shape[0]*images[i].shape[1] != raw_labels[i].shape[0]*raw_labels[i].shape[1]) | (images[i].shape[0] != 512):
            to_remove.append(i)
        
    images = [images[i] for i in range(len(images)) if i not in to_remove]
    raw_labels = [raw_labels[i] for i in range(len(raw_labels)) if i not in to_remove]
    names = [names[i] for i in range(len(names)) if i not in to_remove]
    
    print('Removed {} images from dataset, {} images remaining.'.format(len(to_remove), len(images)))
    
    return images, raw_labels, names



### PLOT FUNCTIONS
       
def plot_dataset(images,labels,names):
    
    N_lines = len(images) // 3 + 1
    plt.figure(figsize=(18, 5*N_lines))
    
    for i in range(N_lines):
        
        counter_position = 1
        counter_image = 0
        
        while (counter_image < 3) & ((3*i + counter_image) < len(images)):
            
            plt.subplot(N_lines, 6, 6*i + counter_position)
            plt.imshow(images[3*i + counter_image])
            plt.title('Image ' + names[3*i + counter_image], fontsize=15)
            plt.axis('off')
            
            counter_position += 1
            
            plt.subplot(N_lines, 6, 6*i + counter_position)
            plt.imshow(labels[3*i + counter_image])
            plt.title('Labels ' + names[3*i + counter_image], fontsize=15)
            plt.axis('off')
            
            counter_position += 1
            counter_image += 1
            
    plt.tight_layout()
    plt.show()


def plot_features(images, index_viz, feature_functions, N_cols):

    img = images[index_viz]
    N_total = len(feature_functions)
    N_lines = N_total // N_cols + 1
    
    plt.figure(figsize=(2*N_cols, 2.5*N_lines))
    
    for i in range(N_total):
        plt.subplot(N_lines,N_cols,i+1)
        plt.imshow(feature_functions[i](img))
        plt.title(feature_functions[i].__name__, fontsize=15)
        plt.axis('off')
        
    plt.tight_layout()
    plt.show()
    

def plot4(img1,img2,img3,img4,title1,title2,title3,title4):
    
    plt.figure(figsize=(12,6))
    
    plt.subplot(1,4,1)
    plt.imshow(img1)
    plt.title(title1)
    plt.axis('off')
    
    plt.subplot(1,4,2)
    plt.imshow(img2)
    plt.title(title2)
    plt.axis('off')
    
    plt.subplot(1,4,3)
    plt.imshow(img3)
    plt.title(title3)
    plt.axis('off')
    
    plt.subplot(1,4,4)
    plt.imshow(img4)
    plt.title(title4)
    plt.axis('off')
    
    plt.show()
    

def plot_predictions(images, X_train, X_test, y_train, y_test, y_train_label, y_test_label, 
                     y_pred_label_train, y_pred_label, train_ids, test_ids, y_train_imageids, y_test_imageids):
    
    X_train['true_label'] = y_train_label
    X_train['true_color'] = y_train
    X_train['pred_label'] = y_pred_label_train
    X_train['pred_color'] = X_train['pred_label'].map(get_color)
    X_train['ImageId'] = y_train_imageids
    
    X_test['true_label'] = y_test_label
    X_test['true_color'] = y_test
    X_test['pred_label'] = y_pred_label
    X_test['pred_color'] = X_test['pred_label'].map(get_color)
    X_test['ImageId'] = y_test_imageids
          
    print('Predictions for images in test set :')

    for imageid in test_ids:
        img = X_test[X_test['ImageId']==imageid]
        original_image = images[int(imageid)]
        true_image = np.array(img['true_color']).reshape((images[0].shape[0],-1))
        pred_image = np.array(img['pred_color']).reshape((images[0].shape[0],-1))
        smoothed_image = smooth_image(pred_image)
        plot4(original_image,true_image,pred_image,smoothed_image,'Original image','True labels','Predicted labels','Smoothed labels')
        
    print('Predictions for images in train set :')
    
    for imageid in train_ids:
        img = X_train[X_train['ImageId']==imageid]
        original_image = images[int(imageid)]
        true_image = np.array(img['true_color']).reshape((images[0].shape[0],-1))
        pred_image = np.array(img['pred_color']).reshape((images[0].shape[0],-1))
        smoothed_image = smooth_image(pred_image)
        plot4(original_image,true_image,pred_image,smoothed_image,'Original image','True labels','Predicted labels','Smoothed labels')
    
   

### LABELING FUNCTIONS
    
def label_image(img):
    
    img_label = np.zeros((img.shape[0], img.shape[1]))
    
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i,j,0] >= 215) & (img[i,j,0] <= 255) & (img[i,j,1] >= 0) & (img[i,j,1] <= 40) & (img[i,j,2] >= 0) & (img[i,j,2] <= 40):
                img_label[i,j] = 0.05 # background
            elif (img[i,j,0] >= 0) & (img[i,j,0] <= 40) & (img[i,j,1] >= 0) & (img[i,j,1] <= 40) & (img[i,j,2] >= 215) & (img[i,j,2] <= 255):
                img_label[i,j] = 0.35 # eyes
            elif (img[i,j,0] >= 0) & (img[i,j,0] <= 40) & (img[i,j,1] >= 215) & (img[i,j,1] <= 255) & (img[i,j,2] >= 0) & (img[i,j,2] <= 40):
                img_label[i,j] = 0.5 # mouth 
            elif (img[i,j,0] >= 215) & (img[i,j,0] <= 255) & (img[i,j,1] >= 215) & (img[i,j,1] <= 255) & (img[i,j,2] >= 0) & (img[i,j,2] <= 40):
                img_label[i,j] = 0.7 # skin
            elif (img[i,j,0] >= 0) & (img[i,j,0] <= 40) & (img[i,j,1] >= 215) & (img[i,j,1] <= 255) & (img[i,j,2] >= 215) & (img[i,j,2] <= 255):
                img_label[i,j] = 0.8 # nose
            elif (img[i,j,0] >= 110) & (img[i,j,0] <= 150) & (img[i,j,1] >= 0) & (img[i,j,1] <= 40) & (img[i,j,2] >= 0) & (img[i,j,2] <= 40):
                img_label[i,j] = 0.9 # hairs
                
    return img_label


def extract_label(raw_labels):
    
    print('Extracting labels :')
    print('')
    
    labels = []
    counter = 0
    print('{} / {}'.format(counter,len(raw_labels)))
    
    for label in raw_labels:
        labels.append(label_image(label))
        counter += 1
        print('{} / {}'.format(counter,len(raw_labels)))
        
    return labels


def get_label(x):
    if x == 0.05:
        return 'Background'
    if x == 0.35:
        return 'Eyes'
    if x == 0.5:
        return 'Mouth'
    if x == 0.7:
        return 'Skin'
    if x == 0.8:
        return 'Nose'
    if x == 0.9:
        return 'Hair'
    else:
        return 'Unknown'

     
def get_color(x):
    if x == 'Background':
        return 0.05
    if x == 'Eyes':
        return 0.35
    if x == 'Mouth':
        return 0.5
    if x == 'Skin':
        return 0.7
    if x == 'Nose':
        return 0.8
    if x == 'Hair':
        return 0.9
    else:
        return 0


def smooth_image(image):
    return filters.median_filter(image,size=8)
    


### TARGET DEFINITION

def target_definition(X,images,labels):
    
    cumulated_dim = 0
    for i in range(len(images)):
        local_dim = images[i].shape[0] * images[i].shape[1]
        X.loc[cumulated_dim:cumulated_dim+local_dim-1, 'y'] = list(labels[i].ravel().reshape(-1,1))
        cumulated_dim = cumulated_dim + local_dim
        
    X['label'] = X['y'].map(get_label)
    
    return X



### TRAINING FUNCTIONS

def train_test_split(X,ratio):

    all_ids = list(X['ImageId'].unique())
    train_ids = all_ids[:int(ratio*len(all_ids))]
    test_ids = [ids for ids in all_ids if ids not in train_ids]
    
    X_train = X.loc[X['ImageId'].isin(train_ids)].copy()
    X_test = X.loc[X['ImageId'].isin(test_ids)].copy()
    
    y_train = list(X_train['y'])
    y_test = list(X_test['y'])
    
    y_train_imageids = list(X_train['ImageId'])
    y_test_imageids = list(X_test['ImageId'])
    
    y_train_label = list(X_train['label'])
    y_test_label = list(X_test['label'])
    
    X_train.drop(['label','y','ImageId'], axis=1, inplace=True)
    X_test.drop(['label','y','ImageId'], axis=1, inplace=True)
    
    features = list(X_train.columns)
    
    return X_train, X_test, y_train, y_test, y_train_label, y_test_label, train_ids, test_ids, y_train_imageids, y_test_imageids, features


def train(X_train, y_train_label, model, grid_search, params):
    
    if grid_search == True:

        GS = GridSearchCV(model, params, cv=5, n_jobs=-1)
        GS.fit(X_train, y_train_label)
        model = GS.best_estimator_

        print('Best parameters :')
        print(GS.best_params_)

        print('') 
        print('Cross-Validated accuracy : %.3f +/- %.3f' % \
              (-GS.cv_results_['mean_test_score'][GS.best_index_],GS.cv_results_['std_test_score'][GS.best_index_]))
              
    else:
        model.fit(X_train, y_train_label)

    return model
    
    
def plot_confusion_matrix(model, y_true, y_pred, normalize):

    cm = confusion_matrix(y_true, y_pred)
    if normalize==True:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(12,6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    labels = list(model.classes_)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=labels, yticklabels=labels,
           title='Confusion matrix',
           ylabel='True label',
           xlabel='Predicted label')

    fmt = '.2f' if normalize==True else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j,i,format(cm[i,j],fmt),ha="center", va="center",color="white" if cm[i,j]>thresh else "black")
    fig.tight_layout()
    
    np.set_printoptions(precision=2)
    plt.show()
    

def plot_feature_importance(model, features, max_features):
    
    feature_importances = pd.DataFrame(model.feature_importances_,index = features,columns=['importance']).sort_values('importance', ascending=True)
    
    if max_features == None:
        top = feature_importances.iloc[0:]
    else:
        top = feature_importances.iloc[0:max_features]
        
    top.plot(kind='barh',figsize = (12,10))
    plt.legend().remove()
    plt.xticks([])
    plt.title('Feature importance')
    plt.show()