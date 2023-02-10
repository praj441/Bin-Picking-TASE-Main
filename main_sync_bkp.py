#!/usr/bin/env python
import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3,PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool,Header, ColorRGBA
import moveit_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from time import sleep
import time
from tf.transformations import quaternion_from_euler
import tf
import math
import cv2, cv_bridge
from grasp_planning import run_grasp_algo
import os
import threading

from disperse_push_action_final import disperse_task
from point_cloud.srv import point_cloud_service
from declutter_robot_action import Declutter_Action
from vs_rotation_conversion import rotationMatrixToEulerAngles,eulerAnglesToRotationMatrix
from gripper_wsg import Gripper



class ur10_grasp:
	def __init__(self):
		#gripper parameters
		self.gripper = Gripper()
		self.gripper_length = 0.18
		self.gripper_homing_value = 100
		

		if len(sys.argv) > 1:
			self.exp_num = int(sys.argv[1])
		else:
			self.exp_num = 0

		if len(sys.argv) > 2:
			self.total_attempt_current_run = int(sys.argv[2])
		else:
			self.total_attempt_current_run = 0

		rospy.init_node("ur10_mp", anonymous=False)
		rospy.loginfo("hi, is this the init ?: yes")
		rospy.loginfo("Starting node moveit_cartesian_path")
		rospy.on_shutdown(self.cleanup)
		# Initialize the move_group API
		moveit_commander.roscpp_initialize(sys.argv)
		# Initialize the move group for the ur5_arm
		self.arm = moveit_commander.MoveGroupCommander('manipulator')
		# Get the name of the end-effector link
		self.end_effector_link = self.arm.get_end_effector_link()
		# Set the reference frame for pose targets
		reference_frame = "/world"
		# Set the ur5_arm reference frame accordingly
		self.arm.set_pose_reference_frame(reference_frame)
		# Allow replanning to increase the odds of a solution
		self.arm.allow_replanning(True)
		# Allow some leeway in position (meters) and orientation (radians)
		self.arm.set_goal_position_tolerance(0.005)
		self.arm.set_goal_orientation_tolerance(0.01)
		self.arm.set_planning_time(0.1)
		self.arm.set_max_acceleration_scaling_factor(.25)
		self.arm.set_max_velocity_scaling_factor(.15)

		# rostopics
		self.marker_pub = rospy.Publisher('visualization_marker',Marker,queue_size = 100)
		self.normal_pose_publisher = rospy.Publisher('/normal_pose', PoseStamped, queue_size=1)
		
		self.declutter_action = Declutter_Action(self)

		self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.action = None # to be filled in the grasp planning process

		self.main()
		# thread = threading.Thread(target=self.main, args=())
		# thread.start()
		# print('started the thread for main function')
		print('init_end')


	def main(self):
		# first time grasp planning
		from grasp_planning_server_node import Grasp_Planner
		GP = Grasp_Planner()
		self.gripper.run(self.gripper_homing_value)
		GP.grasp_planning(self.path)
		while True:

			self.show_point_in_rviz(GP.action[0:3])
			input('If visualized in rviz enter 1')
			# grasp manipulation action by the robot
			#**************************************************************************
			# move above the target               
			pos_error = self.move_gripper_xyz(GP.action)
			time.sleep(0.5)
			# orient the gripper
			self.orient_gripper(GP.action[3])
			time.sleep(1)
			# move down
			print('moving down by:',-(0.1+GP.finger_depth_down)+pos_error)
			input('\nsafety check: enter 1 to continue:')
			self.move_gripper_updown(-(0.1+GP.finger_depth_down)+pos_error) #down
			# close the fingers
			input('if sure press 1 and enter')
			self.gripper.run(GP.gripper_closing)
			self.gripper.run(GP.gripper_closing-10)
			time.sleep(1)

			self.move_to_predefined('start_pose_ur10.txt')
			print('robot moved to start_pose')
			sleep(0.5)

			
			self.exp_num += 1
			self.total_attempt_current_run += 1
			self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
			if not os.path.exists(self.path):
				os.makedirs(self.path)
		
			#subsequent grasp planning in parallel
			grasp_planning_thread = threading.Thread(target=GP.grasp_planning(), args=(GP,self.path,))
			grasp_planning_thread.daemon = True
			print("Main    : before starting thread")
			grasp_planning_thread.start()
			# move up
			# input('\nsafety check: enter 1 to continue')
			# self.move_gripper_updown(0.05)  #up
			# time.sleep(0.5)
			# move to receptacle and drop
			# input('\nsafety check: enter 1 to continue')
			print('in main after thread start')
			self.move_to_predefined('bin_pose.txt')
			self.gripper.run(self.gripper_homing_value)

			grasp_planning_thread.join()
			#move to start point
			# self.move_to_predefined('start_pose_ur10.txt')
			# sleep(1)
			#**************************************************************************

			

				


	def move_gripper_xyz(self,a):
		cpose = Pose()
		cpose.position.x = a[0]
		cpose.position.y = a[1]
		cpose.position.z = a[2]
		normal_pose = PoseStamped()
		normal_pose.header = Header(frame_id='camera_color_optical_frame')
		normal_pose.pose = cpose
		

		wpose = self.transform_pose_to_world(normal_pose)
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		wpose.position.z = wpose.position.z + (self.gripper_length+0.1) # gripper length consideration and some extra margin
		wpose.orientation = ee_pose.orientation
		return self.move_the_robot_new_way(wpose)

	
	def show_point_in_rviz(self,a,other=False):
		pose_temp = np.array(a).copy()
		if other:
			color = ColorRGBA(1.0, 0.0, 0.0, 0.8)
		else:
			color = ColorRGBA(0.0, 1.0, 0.0, 0.8)
		marker = Marker(
			type=Marker.SPHERE,
			id=0,
			lifetime=rospy.Duration(),
			pose=Pose(Point(a[0], a[1], a[2]), Quaternion(0, 0, 0, 1)),
			scale=Vector3(0.01, 0.01, 0.01),
			header=Header(frame_id='camera_color_optical_frame'),
			color=color)
		self.marker_pub.publish(marker)

	def transform_pose_to_world(self,camera_pose):
		tfl = tf.TransformListener()
		from_link = '/camera_color_optical_frame'
		to_link = '/world'
		t = rospy.Time(0)
		tfl.waitForTransform(to_link,from_link,t,rospy.Duration(5))
		if tfl.canTransform(to_link,from_link,t):
		  world_pose = tfl.transformPose(to_link,camera_pose)
		  self.normal_pose_publisher.publish(world_pose)
		  print world_pose
		else:
		  rospy.logerr('Transformation is not possible!')
		return world_pose.pose

	
	def orient_gripper(self,angle_algo):
		
		if angle_algo > 0:
			angle = angle_algo - 1.57
		else:
			angle = angle_algo + 1.57
		self.arm.set_start_state_to_current_state()
		default_joint_states = self.arm.get_current_joint_values()
		default_joint_states[5] += angle
		self.arm.set_joint_value_target(default_joint_states)
		self.arm.set_start_state_to_current_state()
		plan = self.arm.plan()
		self.arm.execute(plan)


   
		
	def move_gripper_updown(self,value):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		wpose = deepcopy(ee_pose)
		wpose.position.z += value

		self.waypoints = []
		self.waypoints.append(deepcopy(wpose))
		self.arm.set_start_state_to_current_state()
		plan, fraction = self.arm.compute_cartesian_path(self.waypoints, 0.02, 0.0, True)
		self.arm.execute(plan)
		print('robot moved')
		_,error_pos,error_orn = self.calculate_error_ee_pose(wpose)
		print('errors',error_pos,error_orn)

	def move_to_predefined(self,joint_states_txt):
		joint_states = self.arm.get_current_joint_values()
		joint_states = np.loadtxt(joint_states_txt, delimiter=',')
		self.arm.set_joint_value_target(joint_states)
		self.arm.set_start_state_to_current_state()
		plan = self.arm.plan()
		self.arm.execute(plan)
		

	def move_the_robot_new_way(self,wpose,loop=True):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		dx = (wpose.position.x - ee_pose.position.x)
		dy = (wpose.position.y - ee_pose.position.y)
		dz = (wpose.position.z - ee_pose.position.z)
		self.waypoints = []
		for i in range(1,31):
			pose = deepcopy(ee_pose)
			pose.position.x = wpose.position.x - dx/i
			pose.position.y = wpose.position.y - dy/i
			pose.position.z = wpose.position.z - dz/i
			self.waypoints.append(deepcopy(pose))
		self.waypoints.append(deepcopy(wpose))

		self.waypoints = []
		self.waypoints.append(deepcopy(wpose))
		wait_for_safety_input = True
		
		while True:
			self.arm.set_start_state_to_current_state()
			plan, fraction = self.arm.compute_cartesian_path(self.waypoints, 0.01, 0.0, True)
			if wait_for_safety_input:
				c = input('Enter some number to continue:')
				wait_for_safety_input = False
				self.arm.execute(plan)
				print('robot moved')
				time.sleep(0.1)
			else:
				self.arm.execute(plan)
				print('robot moved')
			error_depth,error_pos,error_orn = self.calculate_error_ee_pose(wpose)
			if (error_pos < 0.001 and error_orn < 0.01) or not loop:
				print('errors',error_pos,error_orn)
				break
		return error_depth

	def calculate_error_ee_pose(self,wpose):
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose
		position_error = math.sqrt((wpose.position.x - ee_pose.position.x)**2 + (wpose.position.y - ee_pose.position.y)**2 + (wpose.position.z - ee_pose.position.z)**2)
		orn_error = math.sqrt((wpose.orientation.x - ee_pose.orientation.x)**2 + (wpose.orientation.y - ee_pose.orientation.y)**2 + (wpose.orientation.z - ee_pose.orientation.z)**2 + (wpose.orientation.w - ee_pose.orientation.w)**2)
		print('depth positions',wpose.position.x,ee_pose.position.x)
		return (wpose.position.x - ee_pose.position.x),position_error,orn_error

	def image_callback(self,data):
		if not self.image_start:
			try:
				self.image_start = True
				self.cur_image = cv_bridge.CvBridge().imgmsg_to_cv2(data, "bgr8")
				date_string = time.strftime("%Y-%m-%d-%H:%M")
				print('image callback')
				self.image_end = True
			except cv_bridge.CvBridgeError as e:
				print(e)

	def depth_callback(self,data):
		if not self.depth_start:
			try:
				self.depth_start = True
				self.cur_dmap = cv_bridge.CvBridge().imgmsg_to_cv2(data, desired_encoding="passthrough")
				date_string = time.strftime("%Y-%m-%d-%H:%M")
				print('depth callback')
				self.depth_end = True
			except cv_bridge.CvBridgeError as e:
				print(e)

	def cleanup(self):
		rospy.loginfo("Stopping the robot")
		# Stop any current arm movement
		self.arm.stop()
		#Shut down MoveIt! cleanly
		rospy.loginfo("Shutting down Moveit!")
		moveit_commander.roscpp_shutdown()
		moveit_commander.os._exit(0)

mp=ur10_grasp()
rospy.spin()


