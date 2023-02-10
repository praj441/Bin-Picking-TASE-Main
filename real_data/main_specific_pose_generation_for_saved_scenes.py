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
from cluster_specific_annotation_parallel_min_end_2_end import generate_graspability_scores
from utils_gs import Parameters, points_to_pixels_projection

w = 320 #320
h = 240 #240
param = Parameters(w,h)
param.cone_thrs = 60.0


generate_bare_binimum_training_data = False # Setting it False will generate full data
flag_generate_graspability_scores = True

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)


def add_real_world_noise(D):
	pool_num = random.randint(0,225)
	pool_noise = np.loadtxt('../real_data/noise_samples/{0:06d}.txt'.format(pool_num)).astype(np.float32)
	noise_downsample = pool_noise[::2,::2]
	dmap_noised = D+noise_downsample
	return dmap_noised

if __name__ == "__main__":

	
	path = 'temp' # to store temp files
	inp_data_path = 'test_clutter'
	data_path = 'test_clutter_focused_pose'
	obj_path = '/home/prem/ur_grasping_test/src/case_extension_work/urdfs/google-objects-kubric'
	num_scene = 20
	offset = 0
	PI = pi
	index = 0

	median_depth_map = np.loadtxt(inp_data_path+'/median_depth_map.txt')
	
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

	for scene in range(start,end):

		check_path = data_path+'/{0:06d}_done.txt'.format(scene)
		if os.path.exists(check_path):
			continue
		print(scene)
		D = np.loadtxt(inp_data_path+'/{0:06d}_depth_array.txt'.format(scene)).astype(np.float32)
		# S = np.loadtxt(inp_data_path+'/{0:06d}_seg_mask.txt'.format(scene))
		I = cv2.imread(inp_data_path+'/{0:06d}_ref_image.png'.format(scene))
		# c = input('see, what is there?')
		# I,D,S = get_image(roboId,w,h)
		inputs = {'image': I}
		inputs['sim'] = True
		inputs['darray'] = D
		# inputs['seg_mask'] = S
		inputs['param'] = param
		inputs['median_depth_map'] = median_depth_map
		inputs['dump_dir'] = 'temp/{0}'.format(scene)
		inputs['num_dirs'] = 6 # number of directions for sampling grasp poses at a point
		h,w = D.shape
		
		# D = add_real_world_noise(D)
		# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(D),o3d.camera.PinholeCameraIntrinsic(320,240,307.36,307.07,160.0,120.0),depth_scale=1.0)		
		# # o3d.visualization.draw_geometries([pcd])
		# pc_arr = np.asarray(pcd.points).reshape(h,w,3) # (wxh,3)
		# inputs['pc_arr'] = pc_arr


		# dmap_vis = (D / D.max())*255
		# np.savetxt(path+'/depth_array.txt',D)
		# cv2.imwrite(path+'/ref_image.png',I)
		# cv2.imwrite(path+'/depth_image.png',dmap_vis)


		# saving the data sample
		filtered_pc_arr = np.load(inp_data_path+'/{0:06d}_pc.npy'.format(scene))
		np.save(data_path+'/{0:06d}_pc.npy'.format(scene),filtered_pc_arr)
		centroid_pixels = points_to_pixels_projection(filtered_pc_arr)
		# np.save(data_path+'/{0:06d}_label.npy'.format(scene),label)
		N = filtered_pc_arr.shape[0]
		# annotating for graspability
		if flag_generate_graspability_scores:
			M = 1024
			if M > N:
				M = N
			centroid_pixels_kp = np.array(random.sample(list(centroid_pixels), M))
			grasp_quality_score_arr, angle_arr, width_arr = generate_graspability_scores(inputs,centroid_pixels_kp) #same size as darray
			# np.savetxt(data_path+'/{0:06d}_gqs_array.txt'.format(scene),grasp_quality_score_arr)
			np.save(data_path+'/{0:06d}_angle_array.npy'.format(scene),angle_arr)
			# np.savetxt(data_path+'/{0:06d}_width_array.txt'.format(scene),width_arr)
			# print('background',np.count_nonzero(grasp_quality_score_arr==-1))
			# valids = np.count_nonzero(grasp_quality_score_arr>0)
			# print('invalid',M-valids)
			# print('valid',valids)

		np.savetxt(check_path,[1])

		# grasp_quality_score_arr_vis = (grasp_quality_score_arr / grasp_quality_score_arr.max())*255
		# cv2.imwrite(data_path+'/{0:06d}_gqs_image1.png'.format(scene),grasp_quality_score_arr_vis)

		# angle_arr_vis = (angle_arr / angle_arr.max())*255
		# cv2.imwrite(data_path+'/{0:06d}_angle_image.png'.format(scene),angle_arr_vis)

		# width_arr_vis = (width_arr / width_arr.max())*255
		# cv2.imwrite(data_path+'/{0:06d}_width_image.png'.format(scene),width_arr_vis)
		# c = input('dekho aur sikho kuch')
		# pcd.points = o3d.utility.Vector3dVector(label)
		# o3d.visualization.draw_geometries([pcd])
		# print(np.count_nonzero(label),N)
			
		# print(points_idt)
		# run_grasp_algo(I,D,path,final_attempt=False)

		# c = input("enter something to quit")
	# 	p.resetSimulation()
	# p.disconnect()