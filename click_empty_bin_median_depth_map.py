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

def save_median_depth_empty_bin(path,cam):

	
	cam.click_a_depth_sample()
	darray_empty_bin = cam.cur_depth_map
	np.savetxt(path+'/depth_array_empty_bin.txt',darray_empty_bin)
	median_depth_map = medfilt2d(darray_empty_bin,kernel_size=7)
	np.savetxt(path+'/median_depth_map.txt',median_depth_map)

	cam.click_an_image_sample()
	cv2.imwrite(path+'/image_empty_bin.jpg',cam.cur_image)

	return median_depth_map
	print('done!!')

if __name__ == '__main__':
	path='.'
	cam = Camera('temp')
	save_median_depth_empty_bin(path,cam)