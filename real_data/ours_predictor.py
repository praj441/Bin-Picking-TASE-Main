import os
import sys
import numpy as np
from datetime import datetime
import argparse
import importlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import cv2
# from custom_grasp_planning_algorithm import run_grasp_algo
sys.path.append('../commons/')
from utils_cnn import grasp_pose_prediction
from utils_gs import create_directory
import copy
from math import pi
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '../votenet'
sys.path.append(os.path.join(ROOT_DIR, '.'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
# from pytorch_utils import BNMomentumScheduler
from dump_helper_custom import dump_results
# from tf_visualizer import Visualizer as TfVisualizer
import time
from cluster_net import ClusterNet
from loss_helper import get_loss_custom
import pc_util


parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=20, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--max_epoch', type=int, default=180, help='Epoch to run [default: 180]')
parser.add_argument('--batch_size', type=int, default=8, help='Batch Size during training [default: 8]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--weight_decay', type=float, default=0, help='Optimization L2 weight decay [default: 0]')
parser.add_argument('--bn_decay_step', type=int, default=20, help='Period of BN decay (in epochs) [default: 20]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for BN decay [default: 0.5]')
parser.add_argument('--lr_decay_steps', default='80,120,160', help='When to decay the learning rate (in epochs) [default: 80,120,160]')
parser.add_argument('--lr_decay_rates', default='0.1,0.1,0.1', help='Decay rates for lr decay [default: 0.1,0.1,0.1]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
FLAGS = parser.parse_args()

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
BN_DECAY_STEP = FLAGS.bn_decay_step
BN_DECAY_RATE = FLAGS.bn_decay_rate
LR_DECAY_STEPS = [int(x) for x in FLAGS.lr_decay_steps.split(',')]
LR_DECAY_RATES = [float(x) for x in FLAGS.lr_decay_rates.split(',')]
assert(len(LR_DECAY_STEPS)==len(LR_DECAY_RATES))

LOG_DIR = ROOT_DIR + '/log'
# LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'a')
# LOG_FOUT.write(str(FLAGS)+'\n')
def log_string(out_str):
	# LOG_FOUT.write(out_str+'\n')
	# LOG_FOUT.flush()
	print(out_str)

DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
	else DEFAULT_CHECKPOINT_PATH


num_input_channel = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# if torch.cuda.device_count() > 1:
#   log_string("Let's use %d GPUs!" % (torch.cuda.device_count()))
#   # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#   net = nn.DataParallel(net)


#loading pretrained network
# checkpoint = torch.load('demo_files/pretrained_votenet_on_sunrgbd.tar')
# try:
# 	net.module.load_my_state_dict(checkpoint['model_state_dict'])
# except:
# 	net.load_my_state_dict(checkpoint['model_state_dict'])

# Decay Batchnorm momentum from 0.5 to 0.999
# note: pytorch's BN momentum (default 0.1)= 1 - tensorflow's BN momentum
BN_MOMENTUM_INIT = 0.5
BN_MOMENTUM_MAX = 0.001
bn_lbmd = lambda it: max(BN_MOMENTUM_INIT * BN_DECAY_RATE**(int(it / BN_DECAY_STEP)), BN_MOMENTUM_MAX)
# bnm_scheduler = BNMomentumScheduler(net, bn_lambda=bn_lbmd, last_epoch=start_epoch-1)






def preprocess_point_cloud(point_cloud):
	''' Prepare the numpy point cloud (N,3) for forward pass '''
	point_cloud = point_cloud[:,0:3] # do not use color for now
	floor_height = np.percentile(point_cloud[:,2],0.99)
	height = point_cloud[:,2] - floor_height
	point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
	# point_cloud = random_sampling(point_cloud, FLAGS.num_point)
	# pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
	return point_cloud








class Predictor:
	def __init__(self):
		net = ClusterNet(num_proposal=FLAGS.num_target,
			   input_feature_dim=num_input_channel,
			   vote_factor=FLAGS.vote_factor,
			   sampling=FLAGS.cluster_sampling,testing=True)

		net.to(device)
		criterion = get_loss_custom

		# Load the Adam optimizer
		optimizer = optim.Adam(net.parameters(), lr=BASE_LEARNING_RATE, weight_decay=FLAGS.weight_decay)

		# Load checkpoint if there is any
		it = -1 # for the initialize value of `LambdaLR` and `BNMomentumScheduler`
		start_epoch = 0
		if CHECKPOINT_PATH is not None and os.path.isfile(CHECKPOINT_PATH):
			checkpoint = torch.load(CHECKPOINT_PATH)
			net.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			start_epoch = checkpoint['epoch']
			log_string("-> loaded checkpoint %s (epoch: %d)"%(CHECKPOINT_PATH, start_epoch))

		self.net = net

	def predict(self, inputs_np):
		point_cloud_whole = inputs_np['pc_cloud']
		dump_dir = inputs_np['dump_dir']
		self.net.eval() # set model to eval mode (for bn and dp)
		# point_cloud_whole = np.load('../result_dir/filtered_point_cloud_array.npy')
		# image = cv2.imread('../result_dir/current_image.png')
		# darray = np.loadtxt('../result_dir/depth_array.txt')
		# depth_image = Nocv2.imread('../result_dir/depth_image.png')
		# mask = point_cloud_whole[:,6]
		point_cloud = point_cloud_whole[:,0:3]
		# label = np.load('bin_data/val/000090'+'_label.npy')
		# for i in range(point_cloud.shape[0]):
		# 	print(point_cloud[i,:])
		point_cloud = preprocess_point_cloud(point_cloud)

		num_points = 2000
		point_cloud, choices = pc_util.random_sampling(point_cloud, num_points, return_choices=True)
		# label = point_cloud #label[choices,:]
		# mask = mask[choices]

		pc = np.expand_dims(point_cloud.astype(np.float32), 0)
		# vote_label = np.expand_dims(label.astype(np.float32), 0)
		# mask = np.expand_dims(mask.astype(np.float32), 0)
		gt_prposal = 35

		# Forward pass
		inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
		inputs['proposals'] = gt_prposal
		# inputs['vote_label'] = torch.from_numpy(vote_label).to(device)
		# inputs['vote_label_mask'] = torch.from_numpy(mask).to(device)
		st = time.time()
		with torch.no_grad():
			end_points = self.net(inputs)
		print('CNN time:','**************** ',time.time()-st,' ********************')
		end_points['point_clouds'] = inputs['point_clouds']
		# print(end_points['point_clouds'])
		# Compute loss
		# for key in inputs:
		# 	assert(key not in end_points)
		# 	end_points[key] = inputs[key]
		# loss,end_points = criterion(end_points)#, DATASET_CONFIG)
		# print('loss',loss)
		# print(end_points['aggregated_vote_inds'])
		# print(end_points['aggregated_vote_xyz'])
		# print(point_cloud[end_points['aggregated_vote_inds'].cpu()])

		# dump_dir = '../result_dir'
		# if not os.path.exists(dump_dir): os.mkdir(dump_dir)
		create_directory(dump_dir+'/bmaps')
		create_directory(dump_dir+'/directions')
		create_directory(dump_dir+'/grasp_pose_info')
		# st = time.time() 
		# dump_results(end_points, dump_dir, True)
		# print('ply files dump time:','**************** ',time.time()-st,' ********************')
		print('Dumped detection results to folder %s'%(dump_dir))
		print('done')

		# inputs_np = {'image':image}
		# inputs_np['darray'] = darray
		# inputs_np['depth_image'] = depth_image
		# inputs_np['dump_dir'] = dump_dir
		
		st = time.time()
		outputs = grasp_pose_prediction(end_points,inputs_np)
		print('pose_selection','**************** ',time.time()-st,' ********************')

		cluster_image = outputs['cluster_image']
		cluster_image_gt = outputs['cluster_image_gt']
		final_image = outputs['final_image'] 
		grasp_score = outputs['grasp_score']
		image_cnn_pose = outputs['image_cnn_pose']
		grasp_pose_info = outputs['grasp_pose_info']

		cv2.imwrite(dump_dir+'/bmap.jpg',grasp_pose_info['bmap'])
		cv2.imwrite(dump_dir+'/bmap_ws.jpg',grasp_pose_info['bmap_ws'])
		np.savetxt(dump_dir+'/their_idx.txt',[grasp_pose_info['selected_idx']])
		with open(dump_dir+'/invalid_reason.txt', 'w') as f:
			f.write(grasp_pose_info['invalid_reason'])
		# cv2.imwrite(dump_dir+'/clusters_dl.png',cluster_image)

		# cv2.imwrite(dump_dir+'/final_dl.png',final_image)
		return final_image, cluster_image, grasp_score, image_cnn_pose





