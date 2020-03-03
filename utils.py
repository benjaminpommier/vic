import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import pandas as pd
from os import listdir
from os.path import isfile, join
from scipy.ndimage import filters
import pickle

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

def get_gender_dict():
    gender_dict = { 1:'M', 2:'M', 3:'M', 4:'M', 5:'F',
                6:'M', 7:'F', 8:'M', 9:'F', 10:'M',
                11:'F', 12:'F', 13:'M', 14:'M', 15:'M',
                16:'M', 17:'M', 18:'F', 19:'F', 20:'F',
                21:'F', 22:'F', 23:'F', 24:'F', 25:'F',
                26:'M', 27:'M', 28:'M', 29:'F', 30:'F',
                31:'M', 32:'F', 33:'F', 34:'M', 35:'M',
                36:'F', 37:'F', 38:'M', 39:'F', 40:'M',
                41:'M', 42:'F', 43:'M', 44:'F', 45:'F',
                46:'F', 47:'F', 48:'M', 49:'M', 50:'F',
                51:'M', 52:'M', 53:'M', 54:'F', 55:'M',
                56:'F', 57:'M', 58:'F', 59:'M', 60:'M',
                61:'M', 62:'M', 63:'M', 64:'M', 65:'F',
                66:'M', 67:'F', 68:'F', 69:'M', 70:'M',
                71:'F', 72:'F', 73:'F', 74:'M', 75:'M',
                76:'F', 77:'M', 78:'M', 79:'F', 80:'F',
                81:'F', 82:'M', 83:'F', 84:'F', 85:'M',
                86:'M', 87:'F', 88:'M', 89:'M', 90:'M',
                91:'M', 92:'F', 93:'F', 94:'F', 95:'M',
                96:'M', 97:'F', 98:'F', 99:'F', 100:'F',
                101:'M', 102:'M', 103:'F', 104:'M', 105:'M',
                106:'F', 107:'F', 108:'M', 109:'F', 110:'F',
                111:'F', 112:'M', 113:'F', 114:'M', 115:'F',
                116:'M', 117:'F', 118:'M', 119:'M', 120:'M',
                121:'M', 122:'F', 123:'M', 124:'M', 125:'M',
                126:'F', 127:'M', 128:'F', 129:'F', 130:'F',
                131:'F', 132:'F', 133:'M', 134:'M', 135:'F',
                136:'M', 137:'F', 138:'F', 139:'F', 140:'F',
                141:'F', 142:'F', 143:'F', 144:'M', 145:'M',
                146:'M', 147:'F', 148:'M', 149:'M', 150:'F'}
    return gender_dict

def import_gender_dataset(path_images,sub_dir_labeled,dataset_size):
    
    # Importing dataset
    
    names = sorted([img for img in listdir(path_images+sub_dir_labeled) if isfile(join(path_images+sub_dir_labeled, img))], reverse=True)
    names = names[1:dataset_size+1]
    
    gender_dict = get_gender_dict()

    images = [plt.imread(path_images+sub_dir_labeled+name) for name in names]
    raw_labels = [gender_dict[int(name[:3])] for name in names]
    
    return images, raw_labels, names

def find_points(img_label):
    # utils
    N = img_label.shape[0]
    M = img_label.shape[1]
    grad_right = np.vstack([np.arange(M) for i in range(N)])
    grad_left = np.vstack([np.flip(np.arange(M)) for i in range(N)])
    grad_down = np.hstack([np.arange(N).reshape((N,1)) for i in range(M)])
    grad_up = np.hstack([np.flip(np.arange(N).reshape((N,1))) for i in range(M)])
    
    ## mouth
    mouth = np.where(img_label==get_color('Mouth'), 1, 0)
    
    # Right mouth corner
    mouth_right = np.flip(np.unravel_index(np.argmax(mouth * grad_right), img_label.shape))
    mouth_right = (mouth_right[0], mouth_right[1])
    
    # Left mouth corner
    mouth_left = np.flip(np.unravel_index(np.argmax(mouth * grad_left), img_label.shape))
    mouth_left = (mouth_left[0], mouth_left[1])
    
    ## eyes
    eyes = np.where(img_label==get_color('Eyes'), 1, 0)
    
    # Right eye right corner
    eye_right = np.flip(np.unravel_index(np.argmax(eyes * grad_right), img_label.shape))
    eye_right = (eye_right[0], eye_right[1])
    
    # Left eye left corner
    eye_left = np.flip(np.unravel_index(np.argmax(eyes * grad_left), img_label.shape))
    eye_left = (eye_left[0], eye_left[1])
    
    # nose
    nose = np.where(img_label==get_color('Nose'), 1, 0)
    
    nose_up = np.flip(np.unravel_index(np.argmax(nose * grad_up), img_label.shape))[1] # Nose up
    nose_down = np.flip(np.unravel_index(np.argmax(nose * grad_down), img_label.shape))[1] # Nose down
    nose_right = np.flip(np.unravel_index(np.argmax(nose * grad_right), img_label.shape))[0] # Right Nose
    nose_left = np.flip(np.unravel_index(np.argmax(nose * grad_left), img_label.shape))[0] # Left Nose
    
    r1 = max(mouth_left[0] - eye_left[0],0)
    r2 = max(eye_right[0] - mouth_right[0],0)
    if r1 + r2 == 0:
        r1 = 1
        r2 = 1
    x_nose_tip = int(nose_left + r1 ** 2 / (r1 ** 2 + r2 ** 2) * (nose_right - nose_left))
    y_nose_tip = int(nose_up + 0.8 * (nose_down - nose_up))
    
    nose_tip = (x_nose_tip, y_nose_tip)
    
    # chin
    # mouth_x_coordinate_list = [x for x in range(mouth.shape[1]) for y in range(mouth.shape[0]) if mouth[y,x] == 1]
    # int(np.mean(mouth_x_coordinate_list))
    x_chin = int(mouth_left[0] + r1 / (r1 + r2) * (mouth_right[0] - mouth_left[0]))
    y_chin = int((mouth_left[1] + mouth_right[1])/2 * 1.56 - (eye_left[1] + eye_right[1])/2 * 0.56)
    chin = (x_chin, y_chin)
    
    return {'nose_tip':nose_tip, 'chin':chin, 'eye_left':eye_left, 'eye_right':eye_right, 'mouth_left':mouth_left, 'mouth_right':mouth_right}


def find_distances(img_label):
    # utils
    N = img_label.shape[0]
    M = img_label.shape[1]
    grad_right = np.vstack([np.arange(M) for i in range(N)])
    grad_left = np.vstack([np.flip(np.arange(M)) for i in range(N)])
    grad_down = np.hstack([np.arange(N).reshape((N,1)) for i in range(M)])
    grad_up = np.hstack([np.flip(np.arange(N).reshape((N,1))) for i in range(M)])
    
    ## mouth
    mouth = np.where(img_label==get_color('Mouth'), 1, 0)
    
    # mouth corners
    mouth_right = np.flip(np.unravel_index(np.argmax(mouth * grad_right), img_label.shape))
    mouth_left = np.flip(np.unravel_index(np.argmax(mouth * grad_left), img_label.shape))
    mouth_up = np.flip(np.unravel_index(np.argmax(mouth * grad_up), img_label.shape))
    mouth_down = np.flip(np.unravel_index(np.argmax(mouth * grad_down), img_label.shape))
    
    # coordinates
    mouth_right = (mouth_right[0], mouth_right[1])
    mouth_left = (mouth_left[0], mouth_left[1])
    
    # d mouth
    d_mouth_h = mouth_right[0] - mouth_left[0]
    d_mouth_v = mouth_down[1] - mouth_up[1]
    
    ## eyes
    eyes = np.where(img_label==get_color('Eyes'), 1, 0)
    
    # eyes corners
    eye_right = np.flip(np.unravel_index(np.argmax(eyes * grad_right), img_label.shape))
    eye_left = np.flip(np.unravel_index(np.argmax(eyes * grad_left), img_label.shape))
    eye_down = np.flip(np.unravel_index(np.argmax(eyes * grad_down), img_label.shape))
    eye_up = np.flip(np.unravel_index(np.argmax(eyes * grad_up), img_label.shape))
    
    # coordinates
    eye_right = (eye_right[0], eye_right[1])
    eye_left = (eye_left[0], eye_left[1])
    
    # d eye
    d_eye_h = eye_right[0] - eye_left[0]
    d_eye_v = eye_down[1] - eye_up[1]
    
    # nose
    nose = np.where(img_label==get_color('Nose'), 1, 0)
    
    nose_up = np.flip(np.unravel_index(np.argmax(nose * grad_up), img_label.shape)) 
    nose_down = np.flip(np.unravel_index(np.argmax(nose * grad_down), img_label.shape))
    nose_right = np.flip(np.unravel_index(np.argmax(nose * grad_right), img_label.shape)) 
    nose_left = np.flip(np.unravel_index(np.argmax(nose * grad_left), img_label.shape)) 
    
    # d nose
    d_nose_h = nose_right[0] - nose_left[0]
    d_nose_v = nose_down[1] - nose_up[1]
    
    
    ## skin
    skin = np.where(img_label==get_color('Skin'), 1, 0)
    
    skin_up = np.flip(np.unravel_index(np.argmax(skin * grad_up), img_label.shape)) 
    skin_down = np.flip(np.unravel_index(np.argmax(skin * grad_down), img_label.shape)) 
    skin_right = np.flip(np.unravel_index(np.argmax(skin * grad_right), img_label.shape))
    skin_left = np.flip(np.unravel_index(np.argmax(skin * grad_left), img_label.shape)) 
    
    # d skin
    d_skin_h = skin_right[0] - skin_left[0]
    d_skin_v = skin_down[1] - skin_up[1]
    
    ## skin
    hair = np.where(img_label==get_color('Hair'), 1, 0)
    hair_up = np.flip(np.unravel_index(np.argmax(hair * grad_up), img_label.shape))
    
    
    #useful ratios    
    r1 = max(mouth_left[0] - eye_left[0],0)
    r2 = max(eye_right[0] - mouth_right[0],0)
    r3 = max(nose_left[0] - eye_left[0],0)
    r4 = max(eye_right[0] - nose_right[0],0)
    r5 = max(eye_left[0] - skin_left[0],0)
    r6 = max(skin_right[0] - eye_right[0],0)
    
    v1 = max(mouth_down[1] - eye_up[1],0)
    v2 = max(mouth_up[1] - eye_down[1],0)
    v3 = max(nose_down[1] - eye_down[1],0)
    v4 = max(nose_down[1] - eye_up[1],0)
    v5 = max(mouth_down[1] - nose_up[1],0)
    v6 = max(mouth_up[1] - nose_up[1],0)
    v7 = max(skin_up[1] - hair_up[1],0)
    
    return {'d_mouth_h':d_mouth_h, 'd_mouth_v':d_mouth_v, 'd_nose_h':d_nose_h, 'd_nose_v':d_nose_v,
            'd_eye_h':d_eye_h, 'd_eye_v':d_eye_v, 'd_skin_h':d_skin_h, 'd_skin_v':d_skin_v,
            'r1':r1, 'r2':r2, 'r3':r3, 'r4':r4, 'r5':r5, 'r6':r6,
            'v1':v1, 'v2':v2, 'v3':v3, 'v4':v4, 'v5':v5, 'v6':v6, 'v7':v7}

def extract_volumes(img_label):
    # utils
    N = img_label.shape[0]
    M = img_label.shape[1]
    
    ## mouth
    mouth = np.where(img_label==get_color('Mouth'), 1, 0)
    volume_mouth = np.sum(mouth) / (N * M)
    
    ## eyes
    eyes = np.where(img_label==get_color('Eyes'), 1, 0)
    volume_eyes = np.sum(eyes) / (N * M)
    
    # nose
    nose = np.where(img_label==get_color('Nose'), 1, 0)
    volume_nose = np.sum(nose) / (N * M)
        
    # hair
    hair = np.where(img_label==get_color('Hair'), 1, 0)
    volume_hair = np.sum(hair) / (N * M)
    
    # skin
    skin = np.where(img_label==get_color('Skin'), 1, 0)
    volume_skin = np.sum(skin) / (N * M)
    
    return {'volume_mouth':volume_mouth, 'volume_eyes':volume_eyes, 'volume_nose':volume_nose,
            'volume_hair':volume_hair, 'volume_skin':volume_skin}

    


def save_array(path_images, sub_dir_newlabel, labels, names):
    for label, name in zip(labels, names):
        filename = path_images + sub_dir_newlabel + name[:3] + '.npy'
        fileObject = open(filename, 'wb')
        pickle.dump(label, fileObject)
        fileObject.close()
        
def load_array(filename):
    array = np.array([[9999]]) # data for wrong files
    if filename[-3:] == 'npy':
        fileObject = open(filename, 'rb')
        array = pickle.load(fileObject)
        fileObject.close()
    return array

def load_all_array(path_images, sub_dir_newlabel):
    names = sorted([img for img in listdir(path_images+sub_dir_newlabel)], reverse=True)
    labels = {}
    for name in names:
        array = load_array(path_images + sub_dir_newlabel + name)
        if array[0,0] != 9999:
            labels[int(name[:3])] = array
    return labels

        
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



### TRAINING FUNCTION

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