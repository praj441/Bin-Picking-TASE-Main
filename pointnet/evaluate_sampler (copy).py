import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import cv2
from matplotlib.pyplot import imread
# from custom_grasp_planning_algorithm import run_grasp_algo
sys.path.append('../commons/')
# from custom_grasp_planning_algorithm_dense import sample_and_select_best
# from utils_gs import points_to_pixels_projection
# from utils_gs import draw_clusters_into_image
# from utils_gs import draw_contours_around_clusters
# from utils_gs import select_top_N_grasping_points_via_top_cluster_method
# from utils_gs import select_top_N_grasping_points_via_top_points_method
# from custom_grasp_planning_algorithm_dense import grasp_pose_prediction
from utils_cnn import grasp_pose_prediction
from utils_gs import Parameters
from utils_gs import points_to_pixels_projection
from utils_gs import select_top_N_grasping_points_via_top_points_method
from utils_gs import select_top_N_grasping_points_via_distance_sampling
from custom_grasp_planning_algorithm_dense import evaluate_selected_grasp_poses
# from cluster_graspability_annotation_parallel import generate_graspability_scores
import copy

from sklearn.cluster import KMeans

w = 640 #320
h = 480 #240
param = Parameters(w,h)







sampler = 'pointnet2_pcd_rgb'
sample_type = 'sim'
data_type = 'high'

# ['fcn_rgb', 'high', 'sim'],
batch_task = [['fcn_depth', 'high', 'sim'], ['fcn_rgb', 'mid', 'sim'] , ['fcn_rgb', 'high', 'sim'], ['fcn_rgbd','high', 'sim'],  ['pointnet2_pcd_rgb', 'high', 'sim'], ['pointnet2_pcd_rgb','high', 'real'], ['fcn_rgb', 'high', 'real'] , ['fcn_rgbd', 'high', 'real'], ['random', 'mid', 'sim'] , ['random', 'high', 'real'], ['baseline', 'high', 'real'], ['baseline_adaptive', 'mid', 'sim'],  ['baseline_adaptive', 'high', 'real'] ]

batch_task = [['transformer_pcd', 'mid', 'sim']]

# samplers = ['transformer_pcd']

samplers = ['random','baseline','baseline_adaptive','fcn_rgb','fcn_rgbd','fcn_depth', 'pointnet2_pcd', 'pointnet2_pcd_rgb','votenet', 'pointnet2_fcn']

# for sampler, data_type, sample_type in batch_task:
for sampler in samplers:
	data_type = 'mid'
	sample_type = 'real'
	print(sampler, data_type, sample_type)

	if sampler == 'transformer' or sampler == 'transformer_pcd':
		sys.path.append('Stratified_Transformer')
		from util_st import config
		from util_st.data_util import collate_fn, collate_fn_limit
		from model.stratified_transformer import Stratified
		from functools import partial
		import torch_points_kernels as tp
		from bin_data.voxelize import voxelize 
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		MEAN_COLOR_RGB = np.array([0.5,0.5,0.5])
		def get_parser():
			parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
			parser.add_argument('--config', type=str, default='Stratified_Transformer/config/bin/{0}.yaml'.format(sampler), help='config file')
			parser.add_argument('opts', help='see config/s3dis/s3dis_stratified_transformer.yaml for all options', default=None, nargs=argparse.REMAINDER)
			args = parser.parse_args()
			assert args.config is not None
			cfg = config.load_cfg_from_cfg_file(args.config)
			if args.opts is not None:
				cfg = config.merge_cfg_from_list(cfg, args.opts)
			return cfg

		args = get_parser()

		args.patch_size = args.grid_size * args.patch_size
		args.window_size = [args.patch_size * args.window_size * (2**i) for i in range(args.num_layers)]
		args.grid_sizes = [args.patch_size * (2**i) for i in range(args.num_layers)]
		args.quant_sizes = [args.quant_size * (2**i) for i in range(args.num_layers)]

		net = Stratified(args.downsample_scale, args.depths, args.channels, args.num_heads, args.window_size, \
			args.up_k, args.grid_sizes, args.quant_sizes, rel_query=args.rel_query, \
			rel_key=args.rel_key, rel_value=args.rel_value, drop_path_rate=args.drop_path_rate, concat_xyz=args.concat_xyz, num_classes=args.classes, \
			ratio=args.ratio, k=args.k, prev_grid_size=args.grid_size, sigma=1.0, num_layers=args.num_layers, stem_transformer=args.stem_transformer)

		checkpoint = torch.load('log/{0}/checkpoint_{1}.tar'.format(sampler,data_type))
		net.load_state_dict(checkpoint['model_state_dict'])
		net.to(device)


	if sampler == 'fcn_rgb' or sampler == 'fcn_depth' or sampler == 'fcn_rgbd':
		#initials for fcn backbones
		import torchvision
		import torchvision.transforms as tf
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		
		Net = torchvision.models.segmentation.fcn_resnet101(pretrained=True) # Load net
		Net.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
		if sampler == 'fcn_rgbd':
			Net.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
			transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406,0.4), (0.229, 0.224, 0.225,0.2))])
		else:
			transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

		if torch.cuda.device_count() > 1:
		  print("Let's use %d GPUs!" % (torch.cuda.device_count()))
		  Net = torch.nn.DataParallel(Net)
		# if sampler == 'fcn_rgb':
		# 	checkpoint = torch.load('../FCN/mid_rgb_only/2.torch')
		# elif sampler == 'fcn_depth':
		# 	checkpoint = torch.load('../FCN/regress_depth_only/5.torch')
		# else:
		# 	checkpoint = torch.load('../FCN/regress_rgbd/2.torch')

		checkpoint = torch.load('../FCN/{0}_{1}/weights.torch'.format(data_type,sampler))
		Net.load_state_dict(checkpoint)
		Net.to(device)

		max_pool = torch.nn.MaxPool2d(9, padding=0, stride=9,return_indices=True)



	if sampler == 'pointnet2_pcd' or sampler == 'pointnet2_pcd_rgb' or sampler == 'votenet':
		#************ Loading CNN **************************
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		ROOT_DIR = BASE_DIR
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
		CHECKPOINT_PATH = 'log/checkpoint_{0}_{1}.tar'.format('high',sampler)
		checkpoint = torch.load(CHECKPOINT_PATH)
		net.load_state_dict(checkpoint['model_state_dict'])
		print("-> loaded checkpoint %s"%(CHECKPOINT_PATH))
		net.eval()


	if sampler == 'pointnet2_fcn':
		#************ Loading CNN **************************
		BASE_DIR = os.path.dirname(os.path.abspath(__file__))
		ROOT_DIR = BASE_DIR
		sys.path.append('utils')
		sys.path.append('pointnet2')
		sys.path.append('models')
		from cluster_net_cf import ClusterNet
		import torchvision.transforms as tf
		transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
		import pc_util
		num_points = 2000
		num_input_channel = 1
		net = ClusterNet(num_proposal=20,
					   input_feature_dim=num_input_channel,
					   vote_factor=1,
					   sampling='vote_fps',testing=True)
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		net.to(device)
		CHECKPOINT_PATH = 'log/checkpoint_{0}_{1}.tar'.format('high',sampler)
		checkpoint = torch.load(CHECKPOINT_PATH)
		net.load_state_dict(checkpoint['model_state_dict'])
		print("-> loaded checkpoint %s"%(CHECKPOINT_PATH))
		net.eval()
	#***************************************************

	# sample_path = 'bin_data/val/000215'
	from utils_gs import create_directory
	if sample_type == 'sim':
		data_path = 'bin_data/data_{0}/val'.format(data_type)
		median_depth_map = np.loadtxt(data_path+'/../median_depth_map.txt')
		start_sample = 0
		total_samples = 200
		out_path = data_path + '/../sampler_output/' + sampler
		create_directory(out_path)

	else:
		if data_type == 'high':
			data_path = '../real_data/test_data_level_1'
		else:
			data_path = '../real_data/test_data_mid_level'
		median_depth_map = np.loadtxt(data_path+'/median_depth_map.txt')[::2,::2]
		start_sample = 1
		total_samples = 50
		out_path = data_path + '/sampler_output/' + sampler
		create_directory(out_path)

	# sample_path = '../real_data/train/000001'
	avg_time = 0.0
	total_positive = 0
	Nt = 10



	for i in range(start_sample,start_sample+total_samples):
		st = time.time()
		sample_path = data_path + '/{0:06d}'.format(i)
		# point_cloud_whole = np.load(sample_path+'_pc.npy')
		# # import open3d as o3d
		# # pcd = o3d.geometry.PointCloud()
		# # pcd.points = o3d.utility.Vector3dVector(point_cloud_whole[:,:3])
		# # o3d.visualization.draw_geometries([pcd])

		# mask = point_cloud_whole[:,6]
		# point_cloud = point_cloud_whole[:,0:3]
		# label = np.load(sample_path+'_label.npy')
		# gt_prposal = np.load(sample_path+'_num_objects.npy')
		# image = cv2.imread(sample_path+'_ref_image.png')
		# darray = np.loadtxt(sample_path+'_depth_array.txt')
		# depth_image = cv2.imread(sample_path+'_depth_image.png')
		# gqs_map = np.loadtxt(sample_path+'_gqs_array.txt')
		# # angle_map = np.loadtxt(sample_path+'_angle_array.txt')
		# angle_map = np.load(sample_path+'_angle_array.npy')
		# width_map = np.loadtxt(sample_path+'_width_array.txt')
		# print('gt_prposal',gt_prposal)
		
		


		if sampler == 'fcn_rgb' or sampler == 'fcn_depth' or sampler == 'fcn_rgbd':
			if sampler == 'fcn_depth':
				darray = np.loadtxt(sample_path+'_depth_array.txt')[::2,::2]
				darray = darray - darray.min()
				darray = darray/darray.max()
				# print(darray.shape)
				darray = np.tile(darray[:,:,np.newaxis],(1,1,3))
				img = torch.from_numpy((np.transpose(darray, (2, 0, 1)))).float()
			elif sampler == 'fcn_rgb':
				image = imread(sample_path+'_ref_image.png')[::2,::2]
				img = torch.from_numpy((np.transpose(image, (2, 0, 1)))).float()
			else:
				img_file = sample_path+'_ref_image.png'
				img = imread(img_file)[::2,::2]
				darray = np.loadtxt(sample_path+'_depth_array.txt')[::2,::2]
				darray = darray - darray.min()
				darray = darray/darray.max()

				mix_img = np.zeros((img.shape[0],img.shape[1],4))
				mix_img[:,:,0:3] = img[:,:,0:3]
				mix_img[:,:,3] = darray
				img = torch.from_numpy((np.transpose(mix_img, (2, 0, 1)))).float()

			img = transformImg(img).unsqueeze(0)
			with torch.no_grad():
				Pred=Net(img)['out'].squeeze(0)
			
			# max_pool method
			# Pred,mpi = max_pool(Pred)
			# mpi = mpi.squeeze(0).squeeze(0)
			# Pred = Pred.squeeze(0)
			# mpi = mpi.flatten()
			# mpi = np.array(np.unravel_index(mpi.cpu().numpy(), (240,320))).T
			# v, idx = torch.topk(Pred.flatten(), Nt)
			# top_grasp_points = mpi[idx.cpu().numpy()]
			# gqs_score = 100*v.cpu().numpy()
			
			# distance sampling
			Pred = Pred.squeeze(0).cpu().numpy()[::8,::8]
			topN_indices = []
			topN_values = []
			dist_thrs = Pred.shape[1]/30
			while len(topN_indices) < Nt:
				topI = np.array(np.unravel_index(Pred.argmax(), Pred.shape))
				for topJ in topN_indices:
					dist_ij = np.linalg.norm(topI - topJ)
					if dist_ij < dist_thrs:
						Pred[topI[0],topI[1]] = 0.0
						continue
				if Pred[topI[0],topI[1]] > 0.0:
					topN_indices.append(topI)
					topN_values.append(Pred[topI[0],topI[1]])
					Pred[topI[0],topI[1]] = 0.0
			top_grasp_points = 8*np.array(topN_indices)
			gqs_score = 100*np.array(topN_values)



			# top_grasp_points = np.array(np.unravel_index(i.cpu().numpy(), Pred.shape)).T
			# gqs_score = 100*Pred.cpu().numpy()[top_grasp_points[:,0],top_grasp_points[:,1]]

			temp = top_grasp_points[:,0].copy()
			top_grasp_points[:,0] = top_grasp_points[:,1]
			top_grasp_points[:,1] = temp

			avg_time += time.time() - st
			# continue
			print('top_grasp_points',top_grasp_points)
			print('gqs_score',gqs_score)

		if sampler == 'transformer' or sampler == 'transformer_pcd':
			point_cloud = np.load(sample_path+'_pc_complete.npy')[::2,::2].reshape(-1,3)#['pc'] # Nx6

				
			# gqs_map = np.load(sample_path+'_gqs_map.npy').reshape(-1)
			if sample_type == 'sim':
				bin_floor_filter = np.loadtxt(sample_path+'_filter_mask_full.txt')
			else:
				bin_floor_filter = np.load(sample_path+'_filter_mask.npy')[::2,::2]
			ind = np.where(bin_floor_filter)
			indices = np.stack((ind[1],ind[0]),1)

			# temp = top_grasp_points[:,0].copy()
			# top_grasp_points[:,0] = top_grasp_points[:,1]
			# top_grasp_points[:,1] = temp
			if args.use_color:
				color_cloud = cv2.imread(sample_path+'_ref_image.png')[::2,::2].reshape(-1,3)
				color_cloud = (point_cloud/255-MEAN_COLOR_RGB)
			else:
				color_cloud = point_cloud

			bin_floor_filter = bin_floor_filter.reshape(-1).astype(bool)
			coord, feat = point_cloud[bin_floor_filter], color_cloud[bin_floor_filter]

			coord_min = np.min(coord, 0)
			coord -= coord_min
			uniq_idx = voxelize(coord, 0.004)
			coord, feat, indices = coord[uniq_idx], feat[uniq_idx], indices[uniq_idx]

			# processing
			coord_min, coord_max = np.min(coord, 0), np.max(coord, 0)
			coord -= (coord_min + coord_max) / 2.0

			coord = torch.FloatTensor(coord)
			feat = torch.FloatTensor(feat)
			offset = []
			offset.append(coord.shape[0])
			offset = torch.IntTensor(offset)

			offset_ = offset.clone()
			offset_[1:] = offset_[1:] - offset_[:-1]
			batch = torch.cat([torch.tensor([ii]*o) for ii,o in enumerate(offset_)], 0).long()

			sigma = 1.0
			radius = 2.5 * args.grid_size * sigma
			neighbor_idx = tp.ball_query(radius, args.max_num_neighbors, coord, coord, mode="partial_dense", batch_x=batch, batch_y=batch)[0]
		
			coord, feat,  offset = coord.cuda(non_blocking=True), feat.cuda(non_blocking=True),offset.cuda(non_blocking=True)
			batch = batch.cuda(non_blocking=True)
			neighbor_idx = neighbor_idx.cuda(non_blocking=True)
			assert batch.shape[0] == feat.shape[0]
			if args.concat_xyz:
				feat = torch.cat([feat, coord], 1)

			# Forward pass
			output = 100*net(feat, coord, offset, batch, neighbor_idx)[:,0].detach().cpu().numpy()

			top_indices = select_top_N_grasping_points_via_distance_sampling(indices,output, topN=Nt)
			top_grasp_points = indices[top_indices]
			gqs_score = output[top_indices]

			print(top_grasp_points)
			print(gqs_score)
			avg_time += time.time() - st
			# continue



		if sampler == 'pointnet2_pcd' or sampler == 'pointnet2_pcd_rgb' or sampler == 'votenet':
			point_cloud_whole = np.load(sample_path+'_pc.npy')
			if sampler == 'pointnet2_pcd_rgb':
				point_cloud = point_cloud_whole
			else:
				point_cloud = point_cloud_whole[:,0:3]
			point_cloud = pc_util.preprocess_point_cloud(point_cloud)
			num_points = 2000
			point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
			pc = np.expand_dims(point_cloud.astype(np.float32), 0)
			
			# Forward pass
			inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
			
			with torch.no_grad():
				end_points = net(inputs)

			points = end_points['seed_xyz'].detach().cpu().numpy()[0] # (1024,3)
			indices = end_points['aggregated_vote_cluster_inds'].detach().cpu().numpy() # (1024,)
			gqs_score_predicted = end_points['gqs_score'].detach().cpu().numpy()[0]
			gqs_score_predicted = np.where(gqs_score_predicted<0,0,gqs_score_predicted)
			gqs_score_predicted = 100*gqs_score_predicted
			P = points_to_pixels_projection(points,w=param.w,h=param.h,fx=param.f_x,fy=param.f_y)
			top_indices = select_top_N_grasping_points_via_distance_sampling(P,gqs_score_predicted, topN=Nt)
			top_grasp_points = P[top_indices]/2
			gqs_score = gqs_score_predicted[top_indices]
			avg_time += time.time() - st
			# continue
		
		if sampler == 'pointnet2_fcn':
			point_cloud_whole = np.load(sample_path+'_pc.npy')
			point_cloud = point_cloud_whole[:,0:3]
			point_cloud = pc_util.preprocess_point_cloud(point_cloud)
			num_points = 2000
			point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
			pc = np.expand_dims(point_cloud.astype(np.float32), 0)

			darray = np.loadtxt(sample_path+'_depth_array.txt')
			darray = darray - darray.min()
			darray = darray/darray.max()
			# print(darray.shape)
			darray = np.tile(darray[:,:,np.newaxis],(1,1,3))
			img = torch.from_numpy((np.transpose(darray, (2, 0, 1)))).float()
			img = transformImg(img).unsqueeze(0)
			# Forward pass
			inputs = {'point_clouds': torch.from_numpy(pc).to(device),'map_2d':img.to(device)}

			with torch.no_grad():
				end_points = net(inputs)

			points = end_points['seed_xyz'].detach().cpu().numpy()[0] # (1024,3)
			# indices = end_points['aggregated_vote_cluster_inds'].detach().cpu().numpy() # (1024,)
			gqs_score_predicted = end_points['gqs_score'].detach().cpu().numpy()[0]
			gqs_score_predicted = np.where(gqs_score_predicted<0,0,gqs_score_predicted)
			gqs_score_predicted = 100*gqs_score_predicted
			P = points_to_pixels_projection(points,w=param.w,h=param.h,fx=param.f_x,fy=param.f_y)
			top_indices = select_top_N_grasping_points_via_distance_sampling(P,gqs_score_predicted, topN=Nt)
			top_grasp_points = P[top_indices]/2
			gqs_score = gqs_score_predicted[top_indices]
			avg_time += time.time() - st

		if sampler == 'random':
			point_cloud_whole = np.load(sample_path+'_pc.npy')
			point_cloud = point_cloud_whole[:,0:3]
			points = point_cloud[np.random.choice(point_cloud.shape[0], Nt, replace=False), :]
			top_grasp_points = points_to_pixels_projection(points,w=param.w,h=param.h,fx=param.f_x,fy=param.f_y)/2
			gqs_score = np.zeros(Nt)
			avg_time += time.time() - st
			# continue

		if sampler == 'baseline':
			darray = np.loadtxt(sample_path+'_depth_array.txt')[::2,::2]
			centroid_pixels_3D, objectness_ratio = param.median_depth_based_filtering(darray,median_depth_map,0.90)
			kmeans = KMeans(n_clusters=Nt,n_init=6,max_iter=1500)
			kmeans.fit(centroid_pixels_3D)
			label = kmeans.labels_
			centers_3d = kmeans.cluster_centers_

			top_grasp_points = centers_3d[:,0:2]
			gqs_score = np.zeros(Nt)
			avg_time += time.time() - st
			# continue

		if sampler == 'baseline_adaptive':
			darray = np.loadtxt(sample_path+'_depth_array.txt')[::2,::2]
			centroid_pixels_3D, objectness_ratio = param.median_depth_based_filtering(darray,median_depth_map,0.90)
			num_of_clusters = int(135*np.power(objectness_ratio,1.75))
			if num_of_clusters < 10:
				num_of_clusters = 10

			kmeans = KMeans(n_clusters=num_of_clusters,n_init=6,max_iter=1500)
			kmeans.fit(centroid_pixels_3D)
			label = kmeans.labels_
			centers_3d = kmeans.cluster_centers_
			centers = centers_3d[:,0:2]

			top_grasp_points = centers[np.random.choice(centers.shape[0], Nt, replace=False), :]
			gqs_score = np.zeros(Nt)
			avg_time += time.time() - st
			# continue


		image = cv2.imread(sample_path+'_ref_image.png')[::2,::2]
		darray = np.loadtxt(sample_path+'_depth_array.txt')[::2,::2]
		depth_image = cv2.imread(sample_path+'_depth_image.png')[::2,::2]
		
		


		inputs_np = {'image':image}
		inputs_np['darray'] = darray
		inputs_np['depth_image'] = depth_image
		inputs_np['dump_dir'] = None
		inputs_np['param'] = param
		inputs_np['pc_cloud'] = None

		inputs_np['top_grasp_points'] = top_grasp_points
		# inputs_np['angles'] = angles
		inputs_np['final_attempt'] = True
		# inputs_np['param'] = param
		inputs_np['gqs_score'] = gqs_score
		inputs_np['num_dirs'] = 6

		result_list = evaluate_selected_grasp_poses(inputs_np)
		max_score_per_point = result_list[8]

		if sample_type == 'real':
			for k,point in enumerate(top_grasp_points):
				if max_score_per_point[k] > 0:
					color = (255,0,0)
				else:
					color = (0,0,255)
				cv2.circle(image, (int(point[0]), int(point[1])), 3, color, -1)

		# cv2.imwrite(out_path+'/{0}.png'.format(i),image)

		# for k,point in enumerate(indices):
		# 	# if max_score_per_point[k] > 0:
		# 	# 	color = (255,0,0)
		# 	# else:
		# 	color = (0,0,255)
		# 	csize = int(output[k]/10)
		# 	cv2.circle(image, (int(point[0]), int(point[1])), csize, color, -1)

		cv2.imwrite(out_path+'/{0}.png'.format(i),image)

		print('my things', np.count_nonzero(max_score_per_point))
		total_positive += np.count_nonzero(max_score_per_point)

	print('**********************************')
	topN_acc = 100 *total_positive / (total_samples*Nt)
	avg_time = avg_time / total_samples
	print('avg time per sample', avg_time)
	print('top{0}_acc: {1:2.2f}'.format(Nt,topN_acc))
	print('**********************************')

	np.savetxt(out_path+'/stats.txt',[Nt,topN_acc,avg_time], fmt='%2.4f')