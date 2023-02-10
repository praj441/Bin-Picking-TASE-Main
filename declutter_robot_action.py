#!/usr/bin/env python
# from clutter.clutter_decide_action import Declutter_Grasp_Decide_Index
# from clutter.local_clutter_score import Declutter
import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
from geometry_msgs.msg import Twist,Quaternion, Pose, Point, Vector3
import moveit_msgs.msg
from sensor_msgs.msg import Image
from time import sleep
import time
import os

from rospy.numpy_msg import numpy_msg
from rospy_tutorials.msg import Floats
from std_msgs.msg import Bool,Header, ColorRGBA
from tf.transformations import quaternion_from_euler
import tf
import math
from math import *
from vs_rotation_conversion import rotationMatrixToEulerAngles

import cv2, cv_bridge
from sensor_msgs.msg import Image
from scipy.misc import imsave
# from skimage.transform import resize
from matplotlib.pyplot import imread
import matplotlib.pyplot as plt
# from find_grasp_regions_declutter import run_grasp_algo
# from new_gdi_algo import run_grasp_algo
import threading
from visualization_msgs.msg import Marker

from point_cloud.srv import point_cloud_service
from disperse_push_action_final import disperse_task

declutter_tries=0
consecutive_declutter_tries = 0

image_start = False
image_end = False
depth_start = False
depth_end = False
new_msg = False
msg = None
start = True
start_pose_set = True
end = False
error_list = []
setting_list = []
plan = None
cam_wpos = np.zeros(6,)
cur_image = None
cur_dmap = None
dmap = None
image = None
Local_Clutter_List = None

import time
import threading

#gripper parameters
from gripper_wsg import Gripper
gripper_length = 0.17
gripper_homing_value = 100
gripper_grasp_value = 35

# if len(sys.argv) > 1:
# 	exp_num = int(sys.argv[1])
# else:
# 	exp_num = 0

# if len(sys.argv) > 2:
# 	total_attempt_current_run = int(sys.argv[2])
# else:
# 	total_attempt_current_run = 0

def query_point_cloud_client(x, y):
	rospy.wait_for_service('point_cloud_access_service')
	try:
		get_3d_cam_point = rospy.ServiceProxy('point_cloud_access_service', point_cloud_service)
		resp = get_3d_cam_point(np.array([x, y]))
		return resp.cam_point
	except rospy.ServiceException as e:
		print("Service call failed: %s"%e)

class Declutter_Action:
	def __init__(self, robot_controller):
		# rospy.init_node("declutter_action", anonymous=False)
		# rospy.on_shutdown(self.cleanup)

		self.robot_controller = robot_controller
		# print('started the declutter action node')
		# print('init_end')

	def constraint_motion_to_fixed_distance(self,S,E,dc=50.0):
		d = sqrt( (S[0]-E[0])**2 + (S[1]-E[1])**2 )
		m = dc/d
		print('distance',d)
		x = int((1-m)*S[0] + m*E[0])
		y = int((1-m)*S[1] + m*E[1])

		return np.array([x,y])
 

	def actionDeclutter(self,start_point,end_point,darray):
		
		end_point = self.constraint_motion_to_fixed_distance(start_point,end_point,120.0)

		self.robot_controller.gripper.run(5)
		datum_z = 0.575
		start_depth = darray[start_point[1]][start_point[0]]
		start_offset = 0.20 #+ 0.25
		end_depth = datum_z - 0.025 #darray[end_point[1]][end_point[0]]
		end_offset = 0.20
		print('start_depth',start_depth,'end_depth',end_depth)
		start_xyz = query_point_cloud_client(3.2*start_point[0],2.4*start_point[1])
		start_cpose = np.zeros((6,))
		start_cpose[0:2] = start_xyz[0:2]
		start_cpose[2] = start_xyz[2] - start_offset 
		start_wpose = self.calculate_wpose(start_cpose)

		# end_xyz = pixel_to_xyz(end_point[0],end_point[1],dmap)
		end_xyz = query_point_cloud_client(3.2*end_point[0],2.4*end_point[1])
		end_cpose = np.zeros((6,))
		end_cpose[0:2] = end_xyz[0:2]
		end_cpose[2] = end_depth - end_offset 
		end_wpose = self.calculate_wpose(end_cpose)

		#reach robot start point
		self.robot_controller.show_point_in_rviz(np.array(start_xyz))
		input('Is start point allright?')
		self.robot_controller.show_point_in_rviz(np.array(end_xyz),other=True)
		input('Is end point allright?')

		start_wpose_intermediate = deepcopy(start_wpose)
		start_wpose_intermediate[2] = start_wpose_intermediate[2] + 0.15
		self.move_the_robot(start_wpose_intermediate,loop=False)
		self.move_the_robot(start_wpose,loop=False)


		# self.move_the_robot_waypoints(start_wpose,loop=False)
		time.sleep(0.5)
		# input('is it safe to move downwards?')
		# self.robot_controller.move_gripper_updown(-0.25) # down for zig-zag motion
		#reach robot end point
		self.move_the_robot(end_wpose,loop=False)
		time.sleep(0.5)
		input('is everything allright?')
		self.robot_controller.move_gripper_updown(0.1)
		input('is everything allright?')
		#move robot base point
		self.robot_controller.move_to_predefined('start_pose.txt')
		sleep(1)
		self.robot_controller.gripper.run(500)

		print('declutter action completed')


	def calculate_wpose(self,pose):
		pose_temp = pose.copy()
		#To calibrate between real and simulator
		pose[0] = pose_temp[2]
		pose[1] = -pose_temp[0]
		pose[2] = -pose_temp[1]
		pose[3] = pose_temp[5]
		pose[4] = -pose_temp[3]
		pose[5] = -pose_temp[4]
		transformer = tf.TransformerROS()
		tf_listener = tf.TransformListener()
		while 1:
			if tf_listener.frameExists("camera_link") and tf_listener.frameExists("ee_link"):
				t = tf_listener.getLatestCommonTime("camera_link", "ee_link")
				position, quaternion = tf_listener.lookupTransform("ee_link","camera_link", t)
				eTc = transformer.fromTranslationRotation(position,quaternion)
				print('eTc',eTc)
				np.savetxt('eTc.txt',eTc,delimiter=',',fmt='%.5f')
				break
			else:
				continue
				# print('Trying to reach tf!!')
		# eTc = np.linalg.inv(np.loadtxt('eTc.txt', delimiter=','))
		eTc = np.loadtxt('eTc.txt', delimiter=',')

		ee_pose = self.robot_controller.arm.get_current_pose(self.robot_controller.end_effector_link).pose
		# wpose = deepcopy(ee_pose)
		wTe = transformer.fromTranslationRotation((ee_pose.position.x,ee_pose.position.y,ee_pose.position.z),(ee_pose.orientation.x,ee_pose.orientation.y,ee_pose.orientation.z,ee_pose.orientation.w))
		

		
		q = quaternion_from_euler(pose[3],pose[4],pose[5])
		cTp = transformer.fromTranslationRotation(pose[0:3],q)

		eTp = np.matmul(eTc,cTp) # =eTp
		final_T = np.matmul(wTe,eTp)
		#extracting pose from transformation matrix
		final_p = final_T[0:3,3]
		final_eu = rotationMatrixToEulerAngles(final_T[0:3,0:3])
		return final_p

	def move_the_robot(self,final_p,loop=True):
		ee_pose = self.robot_controller.arm.get_current_pose(self.robot_controller.end_effector_link).pose
		wpose = deepcopy(ee_pose)
		# final_q = quaternion_from_euler(0.0,0.0,0.0)
		# Inserting the pose values
		wpose.position.x = final_p[0]
		wpose.position.y = final_p[1]
		wpose.position.z = final_p[2]
		# wpose.orientation.x = final_q[0]
		# wpose.orientation.y = final_q[1]
		# wpose.orientation.z = final_q[2]
		# wpose.orientation.w = final_q[3]


		self.waypoints = []
		self.waypoints.append(deepcopy(wpose))
		wait_for_safety_input = True
		while True:
			self.robot_controller.arm.set_start_state_to_current_state()
			plan, fraction = self.robot_controller.arm.compute_cartesian_path(self.waypoints, 0.025, 0.0, True)
			if wait_for_safety_input:
				print('Enter some number to continue:')
				c = input()
				# wait_for_safety_input = False
			self.robot_controller.arm.execute(plan)
			print('robot moved')

			time.sleep(0.1)
			joint_state = np.array(self.robot_controller.arm.get_current_joint_values())
			print(joint_state*(180/math.pi))
			error_pos,error_orn = self.calculate_error_ee_pose(wpose)
			print('errors',error_pos,error_orn)
			# input('enter some number')
			if (error_pos < 0.005 and error_orn < 0.01) or not loop:
				break

	def move_the_robot_waypoints(self,final_p,loop=True):
		ee_pose = self.robot_controller.arm.get_current_pose(self.robot_controller.end_effector_link).pose
		wpose1 = deepcopy(ee_pose)
		wpose1.position.x = final_p[0]
		wpose1.position.y = final_p[1]
		wpose1.position.z = final_p[2] + 0.10

		wpose2 = deepcopy(ee_pose)
		wpose2.position.x = final_p[0]
		wpose2.position.y = final_p[1]
		wpose2.position.z = final_p[2]

		# wpose.orientation.x = final_q[0]
		# wpose.orientation.y = final_q[1]
		# wpose.orientation.z = final_q[2]
		# wpose.orientation.w = final_q[3]


		self.waypoints = []
		self.waypoints.append(deepcopy(wpose1))
		self.waypoints.append(deepcopy(wpose2))
		wait_for_safety_input = True
		while True:
			self.robot_controller.arm.set_start_state_to_current_state()
			plan, fraction = self.robot_controller.arm.compute_cartesian_path(self.waypoints, 0.025, 0.0, True)
			if wait_for_safety_input:
				print('Enter some number to continue:')
				c = input()
				# wait_for_safety_input = False
			self.robot_controller.arm.execute(plan)
			print('robot moved')

			time.sleep(0.1)
			joint_state = np.array(self.robot_controller.arm.get_current_joint_values())
			print(joint_state*(180/math.pi))
			error_pos,error_orn = self.calculate_error_ee_pose(wpose2)
			print('errors',error_pos,error_orn)
			# input('enter some number')
			if (error_pos < 0.005 and error_orn < 0.01) or not loop:
				break

	def calculate_error_ee_pose(self,wpose):
		ee_pose = self.robot_controller.arm.get_current_pose(self.robot_controller.end_effector_link).pose
		print(wpose)
		print(ee_pose)
		position_error = math.sqrt((wpose.position.x - ee_pose.position.x)**2 + (wpose.position.y - ee_pose.position.y)**2 + (wpose.position.z - ee_pose.position.z)**2)
		orn_error = math.sqrt((wpose.orientation.x - ee_pose.orientation.x)**2 + (wpose.orientation.y - ee_pose.orientation.y)**2 + (wpose.orientation.z - ee_pose.orientation.z)**2 + (wpose.orientation.w - ee_pose.orientation.w)**2)
		return position_error,orn_error
