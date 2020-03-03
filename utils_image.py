# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:41:48 2020

@author: Florian Bettini
"""
import numpy as np
import cv2

def label_image(img):
    img_label = np.zeros((img.shape[0], img.shape[1]))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if (img[i,j,0] == 255) & (img[i,j,1] == 0) & (img[i,j,2] == 0):
                img_label[i,j] = 0.4 # background
            elif (img[i,j,0] == 0) & (img[i,j,1] == 0) & (img[i,j,2] == 255):
                img_label[i,j] = 0.5 # eyes
            elif (img[i,j,0] == 0) & (img[i,j,1] == 255) & (img[i,j,2] == 0):
                img_label[i,j] = 0.6 # mouth 
            elif (img[i,j,0] == 255) & (img[i,j,1] == 255) & (img[i,j,2] == 0):
                img_label[i,j] = 0.7 # skin
            elif (img[i,j,0] == 0) & (img[i,j,1] == 255) & (img[i,j,2] == 255):
                img_label[i,j] = 0.8 # nose
            elif (img[i,j,0] == 127) & (img[i,j,1] == 0) & (img[i,j,2] == 0):
                img_label[i,j] = 0.9 # hairs
                
    return img_label

def find_points(img_label):
    # utils
    N = img_label.shape[0]
    M = img_label.shape[1]
    grad_right = np.vstack([np.arange(M) for i in range(N)])
    grad_left = np.vstack([np.flip(np.arange(M)) for i in range(N)])
    grad_down = np.hstack([np.arange(N).reshape((N,1)) for i in range(M)])
    grad_up = np.hstack([np.flip(np.arange(N).reshape((N,1))) for i in range(M)])
    
    ## mouth
    mouth = np.where(img_label==0.6, 1, 0)
    
    # Right mouth corner
    mouth_right = np.flip(np.unravel_index(np.argmax(mouth * grad_right), img_label.shape))
    
    # Left mouth corner
    mouth_left = np.flip(np.unravel_index(np.argmax(mouth * grad_left), img_label.shape))
    
    ## eyes
    eyes = np.where(img_label==0.5, 1, 0)
    
    # Right eye right corner
    eye_right = np.flip(np.unravel_index(np.argmax(eyes * grad_right), img_label.shape))
    
    # Left eye left corner
    eye_left = np.flip(np.unravel_index(np.argmax(eyes * grad_left), img_label.shape))
    
    # nose
    nose = np.where(img_label==0.8, 1, 0)
    
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
    
    return nose_tip, chin, eye_left, eye_right, mouth_left, mouth_right

def head_pose(im, coordinates):
    # Read Image
    size = im.shape
         
    #2D image points. If you change the image, you need to change vector
    image_points = np.array([
                                coordinates[0],     # Nose tip
                                coordinates[1],     # Chin
                                coordinates[2],     # Left eye left corner
                                coordinates[3],     # Right eye right corner
                                coordinates[4],     # Left Mouth corner
                                coordinates[5]      # Right mouth corner
                            ], dtype="double")
     
    # 3D model points.
    model_points = np.array([
                                (0.0, 0.0, 0.0),             # Nose tip
                                (0.0, -330.0, 0.0),        # Chin
                                (-225.0, 170.0, 0.0),     # Left eye left corner
                                (225.0, 170.0, 0.0),      # Right eye right corne
                                (-150.0, -150.0, 0.0),    # Left Mouth corner
                                (150.0, -150.0, 0.0)      # Right mouth corner 
                            ])
     
    
    # Camera internals
     
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                             [[focal_length, 0, center[0]],
                             [0, focal_length, center[1]],
                             [0, 0, 1]], dtype = "double"
                             )

     
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
     
    
    # Project a 3D point (0, 0, 1000.0) onto the image plane
    # We use this to draw a line sticking out of the nose
    
    
    nose_end_point2D, jacobian = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
    
    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
    
    
    p1 = ( int(image_points[0][0]), int(image_points[0][1]))
    p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
    
    im = cv2.line(im, p1, p2, (255,0,0), 2)
    
    return im