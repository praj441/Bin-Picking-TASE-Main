import time
import random
import sys
from PIL import Image
import numpy as np
from math import *
import cv2
from tqdm import tqdm
from os import path
from datetime import datetime
import os
import logging
import open3d as o3d
import sys
import torch
sys.path.append('../commons/')
from utils_gs import Parameters
from utils_gs import points_to_pixels_projection
from utils_gs import select_top_N_grasping_points_via_top_points_method
from custom_grasp_planning_algorithm_dense import evaluate_selected_grasp_poses_parallel

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

w = 320 #320
h = 240 #240
param = Parameters(w,h)
param.cone_thrs = 45.0


generate_bare_binimum_training_data = False # Setting it False will generate full data
flag_generate_graspability_scores = False

manualSeed = random.randint(1, 10000)  # fix seed
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
np.random.seed(manualSeed)

#************ Loading CNN **************************
sys.path.append('utils')
sys.path.append('pointnet2')
sys.path.append('models')
from cluster_net import ClusterNet
import pc_util
num_points = 2000
num_input_channel = 1
net = ClusterNet(num_proposal=20,
			   input_feature_dim=num_input_channel,
			   vote_factor=1,
			   sampling='vote_fps',testing=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
CHECKPOINT_PATH = 'log/checkpoint.tar'
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
print("-> loaded checkpoint %s"%(CHECKPOINT_PATH))
net.eval()
#***************************************************

fig = plt.figure()
plt.rcParams['figure.dpi'] = 200
plt.rcParams['savefig.dpi'] = 200
plt.ion()
fig.show()

def add_real_world_noise(D):
	pool_num = random.randint(0,225)
	pool_noise = np.loadtxt('../real_data/noise_samples/{0:06d}.txt'.format(pool_num)).astype(np.float32)
	noise_downsample = pool_noise[::2,::2]
	dmap_noised = D+noise_downsample
	return dmap_noised

if __name__ == "__main__":

	
	path = 'temp' # to store temp files
	inp_data_path = '../real_data/test_data_level_1'
	data_path = '../votenet/bin_data/data_gp_cls/test'
	num_scene = 2
	offset = 0
	PI = pi
	index = 0

	median_depth_map = np.loadtxt(inp_data_path+'/../median_depth_map.txt')
	st = time.time()

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

	for scene in range(start,end):
		check_path = data_path+'/{0:06d}_done.txt'.format(scene)
		if os.path.exists(check_path):
			continue
		
		#********** CNN processing ****************************
		sample_path = inp_data_path+'/{0:06d}'.format(scene)
		point_cloud = np.load(sample_path+'_pc.npy')[:,0:3]
		point_cloud = pc_util.preprocess_point_cloud(point_cloud)
		point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
		pc = np.expand_dims(point_cloud.astype(np.float32), 0)
		inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
		with torch.no_grad():
			end_points = net(inputs)

		points = end_points['seed_xyz'].detach().cpu().numpy()[0] # (1024,3)
		indices = end_points['aggregated_vote_cluster_inds'].detach().cpu().numpy() # (1024,)
		gqs_score_predicted = end_points['gqs_score'].detach().cpu().numpy()[0]
		gqs_score_predicted = np.where(gqs_score_predicted<0,0,gqs_score_predicted)
		gqs_score_predicted = 100*gqs_score_predicted
		P = points_to_pixels_projection(points,w=param.w,h=param.h,fx=param.f_x,fy=param.f_y)
		top_indices = select_top_N_grasping_points_via_top_points_method(P,gqs_score_predicted, topN=10)
		top_grasp_points = P[top_indices]
		gqs_score = gqs_score_predicted[top_indices]
		#*********************************************************

		#************ Analytical reasoning processing ******************************8
		D = np.loadtxt(inp_data_path+'/{0:06d}_depth_array.txt'.format(scene)).astype(np.float32)
		# S = np.loadtxt(inp_data_path+'/{0:06d}_seg_mask.txt'.format(scene))
		I = cv2.imread(inp_data_path+'/{0:06d}_ref_image.png'.format(scene))
		pc_arr = np.load(sample_path+'_pc_complete.npy').astype(np.float32).reshape(-1,3)

		pcd = o3d.geometry.PointCloud()
		pcd.points = o3d.utility.Vector3dVector(pc_arr)
		o3d.visualization.draw_geometries([pcd])

		pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(D),o3d.camera.PinholeCameraIntrinsic(320,240,307.36,307.07,160.0,120.0),depth_scale=1.0)		
		o3d.visualization.draw_geometries([pcd])

		inputs_np = {'image':I}
		inputs_np['darray'] = D
		inputs_np['depth_image'] = None
		inputs_np['dump_dir'] = None
		inputs_np['seg_mask'] = None
		inputs_np['param'] = param
		inputs_np['pc_arr'] = pc_arr

		inputs_np['top_grasp_points'] = top_grasp_points
		inputs_np['final_attempt'] = True
		inputs_np['gqs_score'] = gqs_score
		inputs_np['num_dirs'] = 6

		inputs_np['slanted_pose_detection'] = False #False
		inputs_np['cone_detection'] = False #False
		GDI_calculator_all_real = evaluate_selected_grasp_poses_parallel(inputs_np)[4]
		#********************************************************************

		#************* Saving the results **********************************
		for i,gc in enumerate(GDI_calculator_all_real):

			cv2.imwrite(data_path+'/{0:06d}_pose_image.jpg'.format(index),gc.final_image)
			np.save(data_path+'/{0:06d}_pcd.npy'.format(index),gc.pmap)

			ref_img = mpimg.imread(data_path+'/{0:06d}_pose_image.jpg'.format(index))
			fig.clf()
			ax = fig.add_subplot(1, 2, 1)
			imgplot = plt.imshow(ref_img)
			ax.set_title('Sample:{0}'.format(index))

			cv2.imwrite(data_path+'/{0:06d}_bmap.jpg'.format(index),gc.bmap)
			ref_img = mpimg.imread(data_path+'/{0:06d}_bmap.jpg'.format(index))
			ax = fig.add_subplot(1, 2, 2)
			imgplot = plt.imshow(ref_img)
			ax.set_title('bmap')

			invalid_id = int(input('label?'))
			np.savetxt(data_path+'/{0:06d}_label.txt'.format(index),[invalid_id])
			print(index,invalid_id)
			index = index + 1

		# terminal file
		# np.savetxt(check_path,[1])