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
from new_gdi_algo_test import run_grasp_algo
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
		self.image_sub = rospy.Subscriber('/camera/color/image_raw', Image, self.image_callback)
		self.depth_sub = rospy.Subscriber('/camera/aligned_depth_to_color/image_raw', Image, self.depth_callback)
		self.marker_pub = rospy.Publisher('visualization_marker',Marker,queue_size = 100)
		self.msg = rospy.wait_for_message('/camera/color/camera_info', CameraInfo)
		self.normal_pose_publisher = rospy.Publisher('/normal_pose', PoseStamped, queue_size=1)
		
		self.declutter_action = Declutter_Action(self)

		self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
		if not os.path.exists(self.path):
			os.makedirs(self.path)

		thread = threading.Thread(target=self.main, args=())
		thread.start()
		print('started the thread for main function')
		print('init_end')


	def main(self):
		while True:
			while not self.image_end or not self.depth_end:
				time.sleep(0.01)
				continue;
		   
			#saving full resolution version
			image = self.cur_image
			dmap = self.cur_dmap.astype(np.float)/1000
			dmap_vis = (dmap / dmap.max())*255
			np.savetxt(self.path+'/depth_array.txt',dmap)
			cv2.imwrite(self.path+'/ref_image.png',image)
			cv2.imwrite(self.path+'/depth_image.png',dmap_vis)
		   
			#******************************* camera alignment correction here *************************
			inverse_cam_transform = None
			# dmap,image, inverse_cam_transform = self.virtual_camera_transformation(dmap,image)

			# grasp planning
			start_time = time.time()
			total_attempt = 3
			final_attempt = False
			for attempt_num in range(total_attempt):
				if attempt_num == 2:
					final_attempt = True
				a,flag,center,valid, boundary_pose, min_depth_difference, pcd = run_grasp_algo(image.copy(),dmap.copy(),self.path,final_attempt=final_attempt)
				if valid:
					break
			if not flag:
				print('error')
				return
			print(a)
			# declutter action if no valid grasp found
			if not valid:
				img = cv2.imread(self.path+'/ref_image.png')
				darray = np.loadtxt(self.path+'/depth_array.txt')
				darray_empty_bin = np.loadtxt(self.path+'/../depth_array_empty_bin.txt')
				start_point,end_point = disperse_task(img,darray,darray_empty_bin,center,self.path)
				np.savetxt(self.path+'/start_point.txt',start_point,fmt='%d')
				np.savetxt(self.path+'/end_point.txt',end_point,fmt='%d')
				self.declutter_action.actionDeclutter(start_point,end_point,darray)

			else:
				gripper_opening = a[4]
				gripper_closing = self.gripper_grasp_value

				# code to be optimized 
				datum_z = 0.575 #0.640
				finger_depth_down = 0.03
				if a[2] > (datum_z-0.042): #0.540:
					finger_depth_down = (datum_z-0.042+finger_depth_down) - a[2]
				if boundary_pose:
					print('boundary_pose',boundary_pose)
					finger_depth_down += -0.004

				print('finger_depth_down:',finger_depth_down)
				print('min_depth_difference:',min_depth_difference)
				
				# if finger_depth_down > min_depth_difference :
				#     finger_depth_down = min_depth_difference
				self.gripper.run(self.gripper_homing_value) #*gripper_opening+5)        

				# if inverse_cam_transform is not None:
				#     a[0:3] = self.affine_transformation(inverse_cam_transform,a[0:3])
				self.show_point_in_rviz(a[0:3])
				input('If visualized in rviz enter 1')
				print('time taken by the algo:{0}'.format(time.time()-start_time))                                                                                                                                                                                                                            
				
				# grasp manipulation action by the robot
				#**************************************************************************
				# move above the target               
				pos_error = self.move_gripper_xyz(a)
				time.sleep(0.5)
				# orient the gripper
				self.orient_gripper(a[3])
				time.sleep(1)
				# move down
				print('moving down by:',-(0.1+finger_depth_down)+pos_error)
				input('\nsafety check: enter 1 to continue:')
				self.move_gripper_updown(-(0.1+finger_depth_down)+pos_error) #down
				# close the fingers
				input('if sure press 1 and enter')
				self.gripper.run(gripper_closing)
				self.gripper.run(gripper_closing-10)
				time.sleep(1)
				# move up
				input('\nsafety check: enter 1 to continue')
				self.move_gripper_updown(0.05)  #up
				time.sleep(0.5)
				# move to receptacle and drop
				input('\nsafety check: enter 1 to continue')
				# self.move_to_predefined('bin_pose.txt')
				self.gripper.run(self.gripper_homing_value)
				#move to start point
				self.move_to_predefined('start_pose_ur10.txt')
				sleep(1)
				#**************************************************************************

			self.image_end = False
			self.depth_end = False
			self.image_start = False
			self.depth_start = False
			self.exp_num += 1
			self.total_attempt_current_run += 1
			self.path = '../images_ce/{0}/{1}'.format(10,self.exp_num)
			if not os.path.exists(self.path):
				os.makedirs(self.path)

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

	def get_point_cloud(self,color, depth, camera_intrinsics, use_mask = True):
		"""
		Given the depth map and intrinsics, returns the point cloud
		"""
		d = depth.copy()
		fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]
		cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
		xmap, ymap = np.arange(d.shape[1]), np.arange(d.shape[0])
		xmap, ymap = np.meshgrid(xmap, ymap)
		points_z = d
		points_x = (xmap - cx) / fx * points_z
		points_y = (ymap - cy) / fy * points_z
		points = np.stack([points_x, points_y, points_z], axis = -1)
		if use_mask:
			mask = (points_z > 0)
			points = points[mask]
		else:
			points = points.reshape((-1, 3))
			c = c.reshape((-1, 3))
		return points

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

	def affine_transformation(self,transform,point):
		point_hm = np.append(point,[1])
		point_trf = np.matmul(transform,point_hm)[0:3]
		return point_trf

	def virtual_camera_transformation(self,dmap,image):
		intrinsics = np.array(self.msg.K).reshape(3, 3)
		cam_matrix = intrinsics.copy()
		print('cam_matrix',cam_matrix)
		np.savetxt('ur10_cam_intrinsics.txt',cam_matrix)
		pc1 = self.get_point_cloud(image,dmap,cam_matrix)
		N = pc1.shape[0]
		transformer = tf.TransformerROS()
		tf_listener = tf.TransformListener()
		ee_pose = self.arm.get_current_pose(self.end_effector_link).pose 
		wTe = transformer.fromTranslationRotation((ee_pose.position.x,ee_pose.position.y,ee_pose.position.z),(ee_pose.orientation.x,ee_pose.orientation.y,ee_pose.orientation.z,ee_pose.orientation.w))
		print('fk',wTe)
		while 1:
			if tf_listener.frameExists("camera_link") and tf_listener.frameExists("ee_link"):
				t = tf_listener.getLatestCommonTime("camera_link", "ee_link")
				position, quaternion = tf_listener.lookupTransform("ee_link","camera_link", t)
				eTc = transformer.fromTranslationRotation(position,quaternion)
				# print('eTc',eTc)
				np.savetxt('eTc.txt',eTc,delimiter=',',fmt='%.5f')
				break
		eTc = np.loadtxt('eTc.txt', delimiter=',')
		wTc_cur = np.matmul(wTe,eTc)
		wTc = np.loadtxt('wTc_start.txt')
		wRc = wTc[0:3,0:3]
		print('angles',180*rotationMatrixToEulerAngles(wRc)/3.14)
		print('angles',rotationMatrixToEulerAngles(wRc))
		cam_transform = np.matmul(np.linalg.inv(wTc),wTc_cur)
		# Transformation for conversion from camera theoritical frame and real world frame.
		cam_theo_transform = np.array([[0, 0, 1, 0],[-1,0,0,0],[0,-1,0,0],[0,0,0,1]])
		cam_transform = np.matmul(np.matmul(np.linalg.inv(cam_theo_transform),cam_transform),cam_theo_transform)
		inverse_cam_transform = np.linalg.inv(cam_transform)
		print('cam_transform',cam_transform)
		np.savetxt('ur10_test_cam_transform.txt',cam_transform)
		N,_ = np.shape(pc1)
		pc1_hm = np.append(pc1,np.ones((N,1)),axis=1) # Nx4
		pc1_trp_hm = np.transpose(pc1_hm) # 4xN 
		print(N, pc1.shape, pc1_hm.shape, pc1_trp_hm.shape)
		pc2_trp_hm = np.matmul(cam_transform, pc1_trp_hm) #4xN
		pc2_hm = np.transpose(pc2_trp_hm).astype(np.float32) #Nx4
		print('homo',pc2_trp_hm[3,:])
		pc2_trp = pc2_trp_hm[0:3,:] # 3xN
		pc1_trp = pc1_trp_hm[0:3,:] # 3x
		N = pc1_trp.shape[1]
		image_projection = np.matmul(cam_matrix,pc1_trp) #3xN
		Xo = np.divide(image_projection[0,:], image_projection[2,:]).astype(np.int16)
		Yo = np.divide(image_projection[1,:], image_projection[2,:]).astype(np.int16)
		print(Xo.shape)
		print(Xo)
		print(Yo)
		image_projection = np.matmul(cam_matrix,pc2_trp) #3xN
		Xn = np.divide(image_projection[0,:], image_projection[2,:]).astype(np.int16)
		Yn = np.divide(image_projection[1,:], image_projection[2,:]).astype(np.int16)
		Dn = pc2_trp[2,:]
		print('Dn',Dn.shape,Xn.shape)
		mask = (Xn<self.msg.width) & (Xn>=0) & (Yn<self.msg.height) & (Yn>=0)
		Xn = Xn[mask]
		Yn = Yn[mask]
		Xo = Xo[mask]
		Yo = Yo[mask]
		Dn = Dn[mask]
		print(Xn.shape[0],Yn.shape)
		print(Xn)
		print(Yn)
		N = Xn.shape[0]
		image_new = np.zeros(image.shape,dtype=np.uint8)
		dmap_new = np.zeros(dmap.shape,dtype=np.float32)
		print('dmap',dmap.shape,image.shape)
		for i in range(N):
			image_new[Yn[i],Xn[i]] = image[Yo[i],Xo[i]]
			dmap_new[Yn[i],Xn[i]] = Dn[i] #dmap[Yo[i],Xo[i]]
		# w,h = dmap.shape
		# for i in range(h):
		#     for j in range(w):
		#         if dmap_new[j][i] < 0.5:
		#             # print(dmap_new[j][i])
		#             if dmap_new[j][i] == 0.0:
		#                 dmap_new[j][i] = 0.54
		dmap_vis = (dmap_new / dmap_new.max())*255
		np.savetxt(self.path+'/depth_array_align.txt',dmap_new)
		cv2.imwrite(self.path+'/ref_image_align.png',image_new)
		cv2.imwrite(self.path+'/depth_image_align.png',dmap_vis)
		#interpolate zero values
		from scipy.interpolate import griddata
		points = np.where(dmap_new != 0)
		values = dmap_new[points]
		xi = np.where(dmap_new == 0)
		dmap_new[xi] = griddata(points, values, xi, method='nearest')
		points = np.where(image_new != 0)
		values = image_new[points]
		xi = np.where(image_new == 0)
		image_new[xi] = griddata(points, values, xi, method='nearest')
		dmap_vis = (dmap_new / dmap_new.max())*255
		cv2.imwrite(self.path+'/depth_image_align_interp.png',dmap_vis)
		cv2.imwrite(self.path+'/ref_image_align_interp.png',image_new)
		return dmap_new, image_new, inverse_cam_transform

	
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
		print('robot moved to bin')

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
