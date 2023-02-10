#!/usr/bin/env python3

import rospy, sys, numpy as np
# import moveit_commander
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
import os

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

def save_samples_periodically(path,cam,gt_value,num_samples):
	existing_samples = len(list(set([os.path.basename(x)[0:6] \
			for x in os.listdir(path)])))
	for i in range(existing_samples, existing_samples + num_samples):
		cam.click_a_depth_sample()
		darray_empty_bin = cam.cur_depth_map
		nan_filter = (darray_empty_bin==0)
		darray_empty_bin[nan_filter] = gt_value
		noise_map = darray_empty_bin - gt_value
		np.savetxt(path+'/{0:06d}.txt'.format(i),noise_map)

		cam.click_an_image_sample()
		cv2.imwrite('temp'+'/{0:06d}.jpg'.format(i),cam.cur_image)

		print(i,':done')
		time.sleep(3)

if __name__ == '__main__':
	path='real_data/noise_samples'
	gt_value  = 0.64 # bin floor depth from the camera
	num_samples = 100
	cam = Camera('temp')
	save_samples_periodically(path,cam,gt_value,num_samples)