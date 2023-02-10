# import pybullet as p
import time
# import pybullet_data
import random
import sys
from PIL import Image
import numpy as np
# from camera import get_image,cam_pose_to_world_pose
# from utils import load_random_urdf
# from utils import load_shapenet
# from utils import load_shapenet_natural_scale
# from utils import init_robot_and_default_camera
# from utils import sample_camera_displacement
# from utils import relative_ee_pose_to_ee_world_pose
# from utils import objects_picked_succesfully
# from cam_ik import move_eye_camera,accurateIK
# from kuka_vipul import Kuka
from math import *
# from find_grasp_regions_declutter_logging_gdi_plus import send_kinect_points
# from new_gdi_algo3 import send_kinect_points
# from new_gdi_algo_test3_time_analysis import run_grasp_algo as send_kinect_points
import cv2
# from roboaction import SetAction
from tqdm import tqdm
from os import path
from datetime import datetime
# from utils import HiddenPrints
import os
import logging

import open3d as o3d
import sys
sys.path.append('../')
# from grasp_planning_algorithm import run_grasp_algo
import random


sys.path.append('../commons/')
from filter_pixels import depth_filter
from cluster_graspability_annotation_parallel_min_end_2_end import generate_graspability_scores
from utils_gs import Parameters
from utils_gs import points_to_pixels_projection
w = 320 #320
h = 240 #240
param = Parameters(w,h)
param.cone_thrs = 60.0


generate_bare_binimum_training_data = False # Setting it False will generate full data
flag_generate_graspability_scores = False

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)


#*******************************************************************

def add_real_world_noise(D):
	pool_num = random.randint(0,225)
	pool_noise = np.loadtxt('../real_data/noise_samples/{0:06d}.txt'.format(pool_num)).astype(np.float32)
	noise_downsample = pool_noise[::2,::2]
	dmap_noised = D+noise_downsample
	return dmap_noised
#*******************************************************************


if __name__ == "__main__":

	
	path = 'temp' # to store temp files
	inp_data_path0 = '../votenet/bin_data/data/val'
	inp_data_path = 'test_clutter_focused_pose'
	data_path = 'test_clutter_pf_specific/cone'
	obj_path = '/home/prem/ur_grasping_test/src/case_extension_work/urdfs/google-objects-kubric'
	num_scene = 20
	offset = 0
	PI = pi
	index = 0

	# median_depth_map = np.loadtxt(inp_data_path+'/../median_depth_map.txt')
	
	data = []
	sample_dirs = 1
	fix_cluster = False
	FSL_only = False
	CRS_only = False
	pose_refine = True
	center_refine = False

	st = time.time()
	max_num_objects = 40
	num_obj = max_num_objects
	objects_picked = 0
	gdi_old_way_counts = 0
	total_algo_time = 0
	algo_run_count = 0
		# basic env. setting of the simulation
		# try:
	#for batch processing

	if len(sys.argv) > 2:
		part = int(sys.argv[1])
		total_parts = int(sys.argv[2])
		part_scenes = int(num_scene/total_parts)
		start = offset + part*part_scenes
		end = offset + (part+1)*part_scenes
		print(part,total_parts,start,end)
	else:
		start = 1
		end = num_scene

	with open(obj_path+'/short_list.txt') as f:
			object_list = f.readlines()
	total_count = 0
	for scene in range(start,end):

		check_path = data_path+'/{0:06d}_done.txt'.format(scene)
		if os.path.exists(check_path):
			continue
		print(scene)

		# saving the data sample
		point_cloud = np.load(inp_data_path+'/{0:06d}_pc.npy'.format(scene))
		angle_map = np.load(inp_data_path+'/{0:06d}_angle_array.npy'.format(scene))

		# pcd = o3d.geometry.PointCloud()
		# pcd.points = o3d.utility.Vector3dVector(point_cloud[:,0:3])
		# o3d.visualization.draw_geometries([pcd])
		P = points_to_pixels_projection(point_cloud)
		pos_count = 0
		neg_count = 0
		label_path = data_path+'/{0:06d}_labels.npy'.format(scene)
		if os.path.exists(label_path):
			gt_label_arr = np.load(label_path)
		else:
			gt_label_arr = np.zeros(100)
		random_order = np.random.permutation(point_cloud.shape[0])
		for i in tqdm(random_order):
			each_point = point_cloud[i]
			# if pos_count >= 30 and neg_count >= 70:
			# 	break
			x = int(P[i][0])
			y = int(P[i][1])
			angles = angle_map[y][x][:,0]
			org_idx = angle_map[y][x][:,1]
			for j,score in enumerate(angles):
				idx = int(org_idx[j])
				if score == 2:
					focused_points = get_an_instance(point_cloud,j,each_point)
					pose_img = cv2.imread('temp/{0}/poses/{1}_{2}.png'.format(scene,idx,j))
					np.save(data_path+'/{0:06d}_pc.npy'.format(total_count),focused_points)
					cv2.imwrite(data_path+'/{0:06d}_pose.png'.format(total_count),pose_img)
					total_count += 1
		np.savetxt(check_path,[1])
		labels = np.zeros(total_count)
		np.save('all_labels_cone.npy',labels)
		# input('see!')
					# pcd.points = o3d.utility.Vector3dVector(focused_points)
					# o3d.visualization.draw_geometries([pcd])
			# print(focused_points.shape)



