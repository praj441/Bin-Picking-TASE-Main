#!/usr/bin/env python3

import rospy, sys, numpy as np
from copy import deepcopy
from time import sleep
import cv2
import cv_bridge
from sensor_msgs.msg import Image
from std_msgs.msg import Float64MultiArray
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
from math import *
import copy

processing = False
new_msg = False
msg = None
cur_depth = None
cur_image_bgr = None
import time


class Camera:   
	def __init__(self):
		self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
		self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
		print('camera init_done')

	def image_callback(self,data):
		global cur_image_bgr
		global processing
		if not processing:
			try:
				cur_image_bgr = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
			except cv_bridge.CvBridgeError as e:
				print(e)

	def depth_callback(self,data):
		global cur_depth
		global processing
		if not processing:
			try:
				cur_depth = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
			except cv_bridge.CvBridgeError as e:
				print(e)


import sys
# sys.path.append("..")
sys.path.append("../commons")
from general_predictor import Predictor
from utils_gs import final_axis_angle, Parameters, draw_top_N_points, draw_grasp_map, draw_rectified_rect

from termcolor import colored
from multiprocessing.connection import Listener


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--exp_num',type=int,default=0, help='Current exp. number')
# FLAGS = parser.parse_args()
# exp_num = FLAGS.exp_num

grasp_pose_pub = rospy.Publisher('grasp_pose',Float64MultiArray,queue_size=1,latch=False)
# D = np.load('current_depth_map.npy').astype(np.float64)
# D = cur_depth.astype(np.float64)/1000
w = 320
h = 240
param = Parameters(w,h)
param.THRESHOLD2 = 0.02

sampler = 'fcn_depth'
data_type = 'mid'
predictor = Predictor(sampler,data_type,Nt=10)
exp_num = 0
path = 'results_ros_policy'
def handle_grasp_pose_request():
	processing = True
	date_string = time.strftime("%Y-%m-%d-%H:%M:%S")
	exp_num = int(np.loadtxt(path+'/exp_num.txt'))
	print('exp_num',exp_num)
	cur_image_bgr = cv2.imread(path+'/{0}_ref_image.png'.format(exp_num))
	dmap = np.loadtxt(path+'/{0}_depth_array.txt'.format(exp_num))
	inputs_np = {}
	inputs_np['image'] = cv2.resize(cur_image_bgr,(w,h))
	inputs_np['darray'] = cv2.resize(dmap,(w,h))

	# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(inputs_np['darray'].astype(np.float32)),o3d.camera.PinholeCameraIntrinsic(640,480,614.72,614.14,320.0,240.0),depth_scale=1.0)     
	#     # o3d.visualization.draw_geometries([pcd])
	# pc_arr = np.asarray(pcd.points).reshape(h,w,3)


	inputs_np['depth_image'] = None
	inputs_np['dump_dir'] = path+'/{0}'.format(exp_num)
	inputs_np['pc_cloud'] = None
	inputs_np['param'] = param
	inputs_np['pc_arr'] = None
	grasp_pose_info = predictor.predict(inputs_np)
	final_rect_pixel_array = grasp_pose_info['final_pose_rectangle']
	gdi_calc = grasp_pose_info['gdi_calculator']

	if gdi_calc.FLS_score is not None and gdi_calc.CRS_score is not None:
		new_centroid, new_gripper_opening, object_width = gdi_calc.draw_refined_pose(0.8*copy.deepcopy(cur_image_bgr), path+'/{0}_final.png'.format(exp_num), scale = 2, thickness=4)
		cx = new_centroid[0]
		cy = new_centroid[1]
		gripper_opening = (float(new_gripper_opening)/param.gripper_finger_space_max)*param.Max_Gripper_Opening_value
		if gripper_opening > 1.0:
			gripper_opening = 1.0
		flag = True
	else:
		cx = final_rect_pixel_array[0][0]
		cy = final_rect_pixel_array[0][1]
		gripper_opening = 1.0
		flag = False
		draw_rectified_rect(img=0.8*copy.deepcopy(cur_image_bgr), pixel_points=2*final_rect_pixel_array,path=path+'/{0}_final.png'.format(exp_num))
	angle = final_axis_angle(final_rect_pixel_array)
	result = [cx,cy,angle,gripper_opening,flag]
	print('info to be published')
	msg = Float64MultiArray()
	msg.data = result
	grasp_pose_pub.publish(msg)
	processing = False

	top_grasp_points = 2*grasp_pose_info['top_grasp_points']
	top_points_image = 0.8*copy.deepcopy(cur_image_bgr)
	draw_top_N_points(top_grasp_points,top_points_image)
	cv2.imwrite(path+'/{0}_top_N_points.png'.format(exp_num),top_points_image)
	

	grasp_map = 100*grasp_pose_info['Pred']
	grasp_map = np.where(grasp_map < 0 , 0 , gqs_map)
	gmap_image = copy.deepcopy(cur_image_bgr)
	draw_grasp_map(grasp_map,path+'/{0}_grasp_map.png'.format(exp_num))
	# cv2.imwrite(path+'/{0}_grasp_map.png'.format(exp_num),gmap_image)

	exp_num += 1
rospy.init_node('gqcnn_ros_policy_publisher')
# mp=Camera()
time.sleep(1)

print(colored("Ready for the grasp pose service.",'green'))
address = ('localhost', 6004) 
listener = Listener(address, authkey=b'secret password')
conn = listener.accept()

while not rospy.is_shutdown():
	try:
		msg = conn.recv()
	except KeyboardInterrupt:
		print("W: interrupt received, proceeding…")
	if msg == 'close':
		conn.close()
		break
	if int(msg):
		print(colored('received a request for grasp pose','green'))
		handle_grasp_pose_request()
		print(colored('grasp pose published','green'))
