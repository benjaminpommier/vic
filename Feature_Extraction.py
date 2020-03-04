import skimage.segmentation as seg
import skimage.filters as fil
import scipy.ndimage.filters as filt
from scipy import signal
import cv2
import numpy as np
import pandas as pd


### NO TRANSFORMATION FEATURES

def Identity(image):
    return image

def DimensionX(image):
    img = image[:,:,0]
    img = np.array([[i]*img.shape[1] for i in range(img.shape[0])]).reshape(img.shape) / (img.shape[0]-1)
    return img

def DimensionY(image):
    img = image[:,:,0]
    img = np.array(list(range(img.shape[1]))*img.shape[0]).reshape(img.shape) / (img.shape[1]-1)
    return img


### SEGMENTATION FEATURES

def SLIC2(image):
    return seg.slic(image,n_segments=2)

def SLIC4(image):
    return seg.slic(image,n_segments=4)

def SLIC20(image):
    return seg.slic(image,n_segments=20)

def SLIC40(image):
    return seg.slic(image,n_segments=40)

def SLIC60(image):
    return seg.slic(image,n_segments=60)

def FW_200_5(image):
    return seg.felzenszwalb(image, scale=200, sigma=0.5, min_size=100)

def FW_150_5(image):
    return seg.felzenszwalb(image, scale=150, sigma=0.5, min_size=100)

def FW_100_5(image):
    return seg.felzenszwalb(image, scale=100, sigma=0.5, min_size=100)

def FW_200_10(image):
    return seg.felzenszwalb(image, scale=200, sigma=1, min_size=100)

def FW_150_10(image):
    return seg.felzenszwalb(image, scale=150, sigma=1, min_size=100)

def FW_100_10(image):
    return seg.felzenszwalb(image, scale=100, sigma=1, min_size=100)

def FW_200_20(image):
    return seg.felzenszwalb(image, scale=200, sigma=2, min_size=100)

def FW_150_20(image):
    return seg.felzenszwalb(image, scale=150, sigma=2, min_size=100)

def FW_100_20(image):
    return seg.felzenszwalb(image, scale=100, sigma=2, min_size=100)

def FW_200_5_UP(image):
    N = image.shape[0]
    M = image.shape[1]
    segmentation = seg.felzenszwalb(image, scale=200, sigma=0.5, min_size=100)
    grad_up = np.hstack([np.flip(np.arange(N).reshape((N,1))) for i in range(M)])
    return segmentation * grad_up / grad_up.max()

def ChanVese(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return seg.chan_vese(image, mu=0.05).astype(int)

def RandomWalker(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    markers = np.zeros(image.shape, dtype=np.uint)
    markers[image < 0.3*255] = 1
    markers[image > 0.5*255] = 2
    return seg.random_walker(image, markers, beta=30, mode='bf')


### POOLING FEATURES

def Max_10(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.maximum_filter(image,size=10)

def Max_20(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.maximum_filter(image,size=20)

def Max_30(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.maximum_filter(image,size=30)

def Min_10(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.minimum_filter(image,size=10)

def Min_20(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.minimum_filter(image,size=20)

def Min_30(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.minimum_filter(image,size=30)

def Min_50(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return filt.minimum_filter(image,size=50)
    

### THRESHOLDING FEATURES

def Otsu(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = fil.threshold_otsu(image)
    binary = image > thresh
    return binary

def Isodata(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = fil.threshold_isodata(image)
    binary = image > thresh 
    return binary

def Li(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = fil.threshold_li(image)
    binary = image > thresh
    return binary

def Triangle(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = fil.threshold_triangle(image)
    binary = image > thresh
    return binary

def Yen(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    thresh = fil.threshold_yen(image)
    binary = image > thresh
    return binary


### EDGE DETECTION FEATURES

def CannyEdge(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(image, 50, 150)

def Frangi(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.frangi(image)

def Hessian(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.hessian(image)

def Laplace(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.laplace(image)

def Prewitt(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.prewitt(image)

def PrewittH(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.prewitt_h(image)

def PrewittV(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.prewitt_v(image)

def Roberts(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.roberts(image)

def RobertsNegDiag(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.roberts_neg_diag(image)

def RobertsPosDiag(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.roberts_pos_diag(image)

def Scharr(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.scharr(image)

def ScharrH(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.scharr_h(image)

def ScharrV(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.scharr_v(image)

def Sobel(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.sobel(image)

def SobelH(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.sobel_h(image)

def SobelV(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return fil.sobel_v(image)


### OTHER FEATURES

def GradientMagnitude(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    Sx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Sy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    gx = signal.convolve2d(image,Sx,'same')
    gy = signal.convolve2d(image,Sy,'same')
    return np.sqrt(gx**2 + gy**2)


### FEATURE EXTRACTION

def feature_extraction(images,feature_functions):

    total_dim = sum([img.shape[0] * img.shape[1] for img in images])
    X = pd.DataFrame(index=range(total_dim))
    
    print('Features extracted:')
    print('')
    
    for f in feature_functions:
        
        print('- ' + f.__name__)
        cumulated_dim = 0
        
        for i in range(len(images)):
            
            feature = f(images[i])
            local_dim = feature.shape[0] * feature.shape[1]
            X.loc[cumulated_dim:cumulated_dim+local_dim-1, 'ImageId'] = int(i)
            
            if len(feature.shape) == 2:
                X.loc[cumulated_dim:cumulated_dim+local_dim-1, f.__name__] = list(feature.ravel().reshape(-1,1))
            else:
                for dim in range(3):
                    X.loc[cumulated_dim:cumulated_dim+local_dim-1, f.__name__+'_'+str(dim)] = list(feature[:,:,dim].ravel().reshape(-1,1))
            
            cumulated_dim = cumulated_dim + local_dim
            
    return X


### HOG & HSV FEATURES

def HOG_HSV(path,names,X):
    
    names_hog = [x[1:3] + '.csv' for x in names]
    hog_features = [pd.read_csv(path + x, header=0, names = ['HOG_HSV'+str(i) for i in range(6)]) 
                     for x in names_hog]
    
    df_hog = pd.concat(hog_features, axis=0).reset_index(drop=True)
    X = pd.concat([X,df_hog], axis=1)
    
    return X