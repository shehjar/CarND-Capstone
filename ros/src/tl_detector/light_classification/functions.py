# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 06:52:50 2017

@author: kauls
"""

import cv2
import random
import numpy as np

def resizing(image,shape=(200,66)):
    height,width = image.shape[:2]
    image = image[30:height-25,:,:]
    resize_img = cv2.resize(image,dsize=shape,interpolation= cv2.INTER_AREA)
    return resize_img

def translation_augmentation(image,translate_limit):
    height,width = image.shape[:2]
    #angle_per_pixel_shift = 0.004
    x_trans_value = np.random.uniform(translate_limit[0], translate_limit[1])
    #angle_aug = angle + x_trans_value*angle_per_pixel_shift
    y_trans_value = np.random.uniform(translate_limit[0], translate_limit[1])
    M = np.float32([[1,0,x_trans_value],[0,1,y_trans_value]])
    aug_image = cv2.warpAffine(image,M,(width,height))
    return aug_image

def brightness_augmentation(image, flag = 'uniform'):
    height,width = image.shape[:2]
    HSV_image = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    #h,s,v = cv2.split(HSV_image)
    if flag == 'shadow':
        multiplier = 0.5
    else:
        multiplier = np.random.uniform()+0.25
    HSV_image[:,:,2] = cv2.multiply(HSV_image[:,:,2],np.array([multiplier]))
    return cv2.cvtColor(HSV_image,cv2.COLOR_HSV2RGB)

def shadow_augmentation(image):
    shadow_image = brightness_augmentation(image,'shadow')
    height,width = image.shape[:2]
    pt1 = [random.randint(0,width),0]
    pt2 = [random.randint(0,width),height]
    polygon = np.array([[0,0],pt1,pt2,[0,height]])
    # Create mask template
    mask = image.copy()
    mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
    mask.fill(0)
    if np.random.uniform() > 0.5:
        src1 = image
        src2 = shadow_image
    else:
        src1 = shadow_image
        src2 = image
    # fill polygon in mask
    cv2.fillConvexPoly(mask,polygon,255)
    # create region of interest
    mask_inv = cv2.bitwise_not(mask)
    img_part1 = cv2.bitwise_and(src1,src1,mask=mask_inv)
    img_part2 = cv2.bitwise_and(src2,src2,mask=mask)
    final_img = cv2.add(img_part1,img_part2)
    return final_img

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
 
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
 
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
 
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
 
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

