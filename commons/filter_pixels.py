#! /usr/bin/env python 
from math import *
import os
import numpy as np
import cv2
import copy                  
import time
import sys
import rospy
# from point_cloud.srv import point_cloud_service
import warnings
from utils_gs import Parameters

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()

manualSeed = np.random.randint(1, 10000)  # fix seed
# print("Random Seed: ", manualSeed)
np.random.seed(manualSeed)
cv2.setRNGSeed(manualSeed)




def draw_samples(img,pixels):
    try:
        l,_ = pixels.shape
        for i in range(l):
            cx = int(pixels[i][0])
            cy = int(pixels[i][1])
            cv2.circle(img, (cx, cy), 2, (0, 0, 255), -1)
    except:
        print('no filtered pixels')
    return img


def median_depth_based_filtering(inputs,discard_prob=0.3):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # # Convert the RGB image to HSV
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    img = inputs['image']
    darray = inputs['darray']
    pc_arr = inputs['pc_arr']
    param = inputs['param']
    median_depth_map = inputs['median_depth_map']
    S=inputs['seg_mask']

    filter_mask = np.zeros(darray.shape)

    filtered = []
    filtered_points = []
    mask = ((median_depth_map - darray) > param.THRESHOLD2) &  (darray!=0)
    for i in range(param.w):
        for j in range(param.h):
            if mask[j][i] and np.random.random()> discard_prob:#0.9:
                filter_mask[j][i] = 1
                filtered.append([i,j,darray[j,i]])#,darray[j,i]])#,img[j,i,2]])#,darray[j,i]])
                # filtered_points.append(pc_arr[j,i,:])
                # print(darray[j,i])
                if S is None:
                    filtered_points.append([pc_arr[j,i,0],pc_arr[j,i,1],pc_arr[j,i,2],img[j,i,0],img[j,i,1],img[j,i,2]])
                else:
                    filtered_points.append([pc_arr[j,i,0],pc_arr[j,i,1],pc_arr[j,i,2],img[j,i,0],img[j,i,1],img[j,i,2],S[j,i]])
                # filtered.append([i,j,img[j,i,0],img[j,i,2]])#,darray[j,i]])#,img[j,i,2]])#,darray[j,i]])

    filtered_points = np.array(filtered_points)
    # print('before',filtered_points.shape)
    #removing nan values
    mask = np.isfinite(filtered_points[:,0]) & np.isfinite(filtered_points[:,1]) & np.isfinite(filtered_points[:,2])
    filtered_points = filtered_points[mask]
    # print('after',filtered_points.shape)
    return np.array(filtered), filtered_points, filter_mask


def depth_filter(inputs,discard_prob=0.0):

    
    # darray = cv2.resize(darray,(w,h))
    # img = cv2.resize(img,(w,h))
    # depth_image = cv2.imread(path+'/depth_image.jpg')
    
    new_img = copy.deepcopy(inputs['image'])
    

    # max_samples = 25000
    # sampled_img, pixels_list = generate_samples(num_of_samples=max_samples, img=img, dmap=darray)
    # filtered_pixels,centroid_pixels_3D = send_request_for_pixel_filtering(pixels_list,darray)  # Service Call for Filtered Pixels #
    
    centroid_pixels_3D, filtered_pc_arr, filter_mask = median_depth_based_filtering(inputs,discard_prob=discard_prob)
    centroid_pixels = centroid_pixels_3D[:,0:2]
    
    filtered_img = draw_samples(new_img,centroid_pixels)
    
    # cv2.imwrite(path+'/filtered_pixels.jpg',filtered_img)
    centroid_pixels = np.float64(centroid_pixels)
    
    return filtered_pc_arr, filtered_img, centroid_pixels, filter_mask


if __name__ == "__main__":
    print('nothing to do. Be happy!!')
   