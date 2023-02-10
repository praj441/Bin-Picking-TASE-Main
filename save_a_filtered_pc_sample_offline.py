#!/usr/bin/env python3

import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
import moveit_msgs.msg
from sensor_msgs.msg import Image
from time import sleep


from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool

import cv2, cv_bridge
from sensor_msgs.msg import Image, PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
import pcl
# from scipy.misc import imsave
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from scipy.signal import medfilt2d
# from find_grasp_regions import run_grasp_algo
from filter_pixels import depth_filter
from utils_gs import Parameters

processing = False
processing1 = False
processing2 = False
new_msg = False
msg = None
start = True
start_pose_set = True
end = False
error_list = []
setting_list = []
plan = None
cam_wpos = np.zeros(6,)

import time
import threading
cur_image = None

from camera import Camera

path = 'result_dir'
cam = Camera(path)
w = 320
h = 240
param = Parameters(w,h)

cam.click_a_camera_sample()

darray = cv2.resize(cam.cur_depth_map,(w,h))
img = cv2.resize(cam.cur_image,(w,h))
cur_pc = cv2.resize(cam.cur_pc,(w,h))

cv2.imwrite(path + '/current_image.png',img)
depth_map_vis = (darray/darray.max())*255
cv2.imwrite(path + '/depth_image.png',depth_map_vis)
np.savetxt(path + '/depth_array.txt',darray)
np.save(path + '/point_cloud_array.npy',cur_pc)

median_depth_map = np.loadtxt('median_depth_map_offline.txt')
median_depth_map = cv2.resize(median_depth_map,(w,h))

filtered_pc_arr, filtered_img = depth_filter(img,darray,cur_pc,param,median_depth_map)

np.save(path+'/filtered_point_cloud_array.npy',filtered_pc_arr)
cv2.imwrite(path+'/filtered_pixels.jpg',filtered_img)

print('filtered pc_arr saved. done!!')