#!/usr/bin/env python
import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3,PoseStamped, WrenchStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import Bool,Header, ColorRGBA
import moveit_msgs.msg
from sensor_msgs.msg import Image, CameraInfo
from time import sleep
import time
from tf.transformations import quaternion_from_euler, euler_from_quaternion
import tf
import math
import cv2, cv_bridge
from grasp_planning_algorithm import run_grasp_algo,grasp_planning
import os
import threading

from disperse_push_action_final import disperse_task
from point_cloud.srv import point_cloud_service
from declutter_robot_action import Declutter_Action
from vs_rotation_conversion import rotationMatrixToEulerAngles,eulerAnglesToRotationMatrix
from gripper_wsg import Gripper


class ur10_grasp:
	def __init__(self):
		self.image_start = False
		self.image_end = False
		self.depth_start = False
		self.depth_end = False
		self.cur_image = None
		self.cur_dmap = None
		self.wrench_start = False
		self.wrench_end = False
		self.force_z = None

		#gripper parameters
		self.gripper = Gripper()
		self.gripper_length = 0.18
		self.gripper_homing_value = 100
		self.gripper_grasp_value = 20

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
		# self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
		# self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
		# self.marker_pub = rospy.Publisher('visualization_marker',Marker,queue_size = 100)
		# self.marker_pub_other = rospy.Publisher('visualization_marker_other',Marker,queue_size = 100)
		# self.msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
		# self.normal_pose_publisher = rospy.Publisher('/normal_pose', PoseStamped, queue_size=1)
		# self.world_pose_publisher = rospy.Publisher('/world_pose', PoseStamped, queue_size=1)
		self.wrench = rospy.Subscriber('/wrench', WrenchStamped, self.wrench_callback)
		self.declutter_action = Declutter_Action(self)

		self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
		if not os.path.exists(self.path):
			os.makedirs(self.path)
		self.action = None # to be filled in the grasp planning process
		self.fov_points = None # to be filled in the grasp planning process
		self.wTc = None # to be filled in the main loop , camera_pose_in_world
		self.main()
		# thread = threading.Thread(target=self.main, args=())
		# thread.start()
		# print('started the thread for main function')
		print('init_end')


	def main(self):
		# self.wrench_start = False
		# self.wrench_end = False
		# while not self.wrench_end:
		# 	time.sleep(0.001)
		# q_current = get_actual_joint_positions()
		HOST = "192.168.2.200"
		PORT = 30002

		import socket 
		s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
		s.connect((HOST, PORT))
		time.sleep(2)

		s.send(("movej([1.574,-1.487,1.606,-1.695,-1.578,-0.789],a=1.396,v=1.047)" +"\n").encode('utf8'))
		time.sleep(2)

		# input('press 1 to continue')

		# pose = self.arm.get_current_pose(self.end_effector_link).pose
		# transformer = tf.TransformerROS()
		# T = transformer.fromTranslationRotation((pose.position.x,pose.position.y,pose.position.z),(pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w))
		# t = T[0:3,3]
		# R = T[0:3,0:3]
		# e = rotationMatrixToEulerAngles(R)
		# next_position = [t[0], t[1], t[2]+0.1, e[0], e[1], e[2]]
		
		# while True:
			
		# 	# s.send(("servoj(get_inverse_kin(p{0}), 0, 0, 0.008,0.1, 300)".format(next_position) + "\n").encode())
		# 	s.send(("servoj([1.574,-1.487,1.606,-1.695,-1.578,0.789], 0, 0, 0.008,0.1, 300)" + "\n").encode())



	def move_gripper_for_grasp(self,d):
		div = 2
		

		# for i in range(steps):
		while True:
			step = d/div
			self.move_gripper_updown(step)
			self.wrench_start = False
			self.wrench_end = False
			while not self.wrench_end:
				time.sleep(0.05)
			print('self.force_z',self.force_z)
			if self.force_z > 50:
				self.move_gripper_updown(-3*step)
				break
			if div < 10:
				div = div+3


	def move_gripper_xyz(self,a):
		cpose = Pose()
		cpose.position.x = a[0]
		cpose.position.y = a[1]
		cpose.position.z = a[2]
		normal_pose = PoseStamped()
		normal_pose.header = Header(frame_id='camera_color_optical_frame')
		normal_pose.pose = cpose
		self.normal_pose_publisher.publish(normal_pose)

		wpose = self.transform_pose_to_world(cpose)
		
		wpose.position.z = wpose.position.z + (self.gripper_length+0.05) # gripper length consideration and some extra margin
		wpose.orientation = self.ee_pose.orientation

		world_pose = PoseStamped()
		world_pose.header = Header(frame_id='world')
		world_pose.pose = wpose
		self.world_pose_publisher.publish(world_pose)

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
		if other:
			self.marker_pub_other.publish(marker)
		else:
			self.marker_pub.publish(marker)

	def transform_pose_to_world(self,pose):
		transformer = tf.TransformerROS()
		wpose = deepcopy(pose)
		cTp = transformer.fromTranslationRotation((pose.position.x,pose.position.y,pose.position.z),(pose.orientation.x,pose.orientation.y,pose.orientation.z,pose.orientation.w))
		wTp = np.matmul(self.wTc,cTp)
		final_p = wTp[0:3,3]
		final_q = tf.transformations.quaternion_from_matrix(wTp)

		# Inserting the pose values
		wpose.position.x = final_p[0]
		wpose.position.y = final_p[1]
		wpose.position.z = final_p[2]
		wpose.orientation.x = final_q[1]
		wpose.orientation.y = final_q[2]
		wpose.orientation.z = final_q[3]
		wpose.orientation.w = final_q[0]

		return wpose
		

	# def transform_pose_to_world(self,camera_pose):
	# 	tfl = tf.TransformListener()
	# 	from_link = '/camera_color_optical_frame'
	# 	to_link = '/world'
	# 	t = rospy.Time(0)
	# 	tfl.waitForTransform(to_link,from_link,t,rospy.Duration(5))
	# 	if tfl.canTransform(to_link,from_link,t):
	# 	  world_pose = tfl.transformPose(to_link,camera_pose)
	# 	  self.normal_pose_publisher.publish(world_pose)
	# 	  print world_pose
	# 	else:
	# 	  rospy.logerr('Transformation is not possible!')
	# 	return world_pose.pose

	def save_camera_world_pose(self):
		transformer = tf.TransformerROS()
		tfl = tf.TransformListener()
		from_link = '/camera_color_optical_frame'
		to_link = '/world'
		t = rospy.Time(0)
		tfl.waitForTransform(to_link,from_link,t,rospy.Duration(5))
		if tfl.canTransform(to_link,from_link,t):
			position, quaternion = tfl.lookupTransform(to_link,from_link,t)     
			self.wTc = transformer.fromTranslationRotation(position,quaternion)
		else:
		  rospy.logerr('Transformation is not possible!')

	
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
		for i in range(1,15):
			pose = deepcopy(ee_pose)
			pose.position.x = wpose.position.x - dx/i
			pose.position.y = wpose.position.y - dy/i
			# pose.position.z = wpose.position.z - dz/i
		for i in range(16,31):
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
				time.sleep(2)
			else:
				self.arm.execute(plan)
				print('robot moved')
			error_depth,error_pos,error_orn = self.calculate_error_ee_pose(wpose)
			if (error_pos < 0.001 and error_orn < 0.01) or not loop:
				print('errors',error_pos,error_orn)
				break
			# break
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

	def wrench_callback(self,data):
		if not self.wrench_start:
			self.wrench_start = True
			self.force_z = data.wrench.force.z
			print('force_z',self.force_z)
			self.wrench_end = True
			# try:
				
			# except:
			# 	print('error in wrench topic')

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


