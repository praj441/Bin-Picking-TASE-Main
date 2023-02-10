#!/usr/bin/env python


import rospy, sys, numpy as np
import moveit_commander
from copy import deepcopy
import moveit_msgs.msg


from time import sleep
from tf.transformations import quaternion_from_euler
import time
from std_msgs.msg import Float64MultiArray  

# from urdf_parser_py.urdf import URDF
# from pykdl_utils.kdl_kinematics import KDLKinematics

import sys

#from get_jacobian import Jacobian

class ur5_mp:
	def __init__(self):
		rospy.init_node("ur5_mp", anonymous=False)
		rospy.loginfo("hi, is this the init ?: yes")
		
				
		self.vel_pub = rospy.Publisher('/joint_group_vel_controller/command',Float64MultiArray,queue_size=1,latch=True)
		self.state_change_time = rospy.Time.now()

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
		self.arm.set_goal_position_tolerance(0.001)
		self.arm.set_goal_orientation_tolerance(0.01)
		self.arm.set_planning_time(0.1)
		self.arm.set_max_acceleration_scaling_factor(.5)
		self.arm.set_max_velocity_scaling_factor(.5)

		#For Jacobian calculation
		# robot = URDF.from_parameter_server()
		# kdl_kin = KDLKinematics(robot, 'base_link', 'ee_link')

		


		# Set the internal state to the current state

		self.arm.set_start_state_to_current_state()
		self.default_joint_states = self.arm.get_current_joint_values()
		# np.savetxt('start_pose.txt',self.default_joint_states,fmt='%1.3f')
		# J = kdl_kin.jacobian(self.default_joint_states)
		# print('J',J)
		
		# J1 = Jacobian(self.default_joint_states)
		# print('J1',J1)


		print(self.default_joint_states)
		# self.default_joint_states[0] = -1.6007002035724085	
		# self.default_joint_states[1] = -1.7271001974688929
		# self.default_joint_states[2] = -2.2029998938189905
		# self.default_joint_states[3] = -0.8079999128924769
		# self.default_joint_states[4] = 1.5951000452041626
		# self.default_joint_states[5] =-0.03099996248354131
		if len(sys.argv) < 2:
			print('please provide pose file')
			sys.exit()
		pose_file = sys.argv[1]
		self.default_joint_states = np.loadtxt(pose_file, delimiter=',')
		
		self.arm.set_joint_value_target(self.default_joint_states)
		self.arm.set_start_state_to_current_state()
		plan = self.arm.plan()
		print('Enter some number to continue:')
		c = input()
		self.arm.execute(plan)
		# print(sys.argv(1))


		# JV_msg = Float64MultiArray()
		# # for i in range(10000):
		# Jinv = np.linalg.inv(J)
		# JV = np.array(np.matmul(Jinv,[0.1,0.0,0.0,0.0,0.0,0.0]))
		# print('JV:',JV)

		
		# JV_msg.data = np.array(JV[0]) #[-0.1,-0.1,0.0,0.0,0.0,0.0]
		# self.vel_pub.publish(JV_msg)
		
		# 	self.arm.set_start_state_to_current_state()
		# 	self.default_joint_states = self.arm.get_current_joint_values()

		# 	J = kdl_kin.jacobian(self.default_joint_states)

		#JV_msg.data = [0.0,0.0,0.0,0.0,0.0,0.0]
		# self.vel_pub.publish(JV_msg)
		# print(JV_msg)
		# self.default_joint_states = self.arm.get_current_joint_values()
		# print(self.default_joint_states)
	
		# move the robot to the predefined pose
		# start_pose = self.arm.get_current_pose(self.end_effector_link).pose
		# print(start_pose)
		# wpose = deepcopy(start_pose)
		# wpose.position.x = 0.0675932149637
		# wpose.position.y = -0.941072190356
		# wpose.position.z =  0.574126907345
		# wpose.orientation.x =   -0.408136540571
		# wpose.orientation.y =   0.434161724245
		# wpose.orientation.z =  0.559893391108
		# wpose.orientation.w =  0.631663393322


		# self.waypoints = []
		# self.waypoints.append(deepcopy(wpose))
		# self.arm.set_start_state_to_current_state()
		# plan, fraction = self.arm.compute_cartesian_path(self.waypoints, 0.02, 0.0, True)
		# print('Enter some number to continue:')
		# # print('plan',plan)
		# # print('fraction',fraction)
		# c = input()
		# self.arm.execute(plan)
		print('init_end')


	def cleanup(self):
		rospy.loginfo("Stopping the robot")

		# Stop any current arm movement
		self.arm.stop()

		#Shut down MoveIt! cleanly
		rospy.loginfo("Shutting down Moveit!")
		moveit_commander.roscpp_shutdown()
		moveit_commander.os._exit(0)


mp=ur5_mp()
rospy.loginfo("hi, is this the start")

# rospy.spin()
