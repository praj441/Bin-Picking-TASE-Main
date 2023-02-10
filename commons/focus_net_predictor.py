
import cv2

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
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = '../votenet'
sys.path.append(os.path.join(ROOT_DIR, '.'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'pointnet2'))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from pytorch_utils import BNMomentumScheduler
# from tf_visualizer import Visualizer as TfVisualizer

# from cluster_net import ClusterNet
from focus_net import FocusNet
from loss_helper import get_loss_focus, compute_acc_focus
# from BMC_loss import BMCLoss
from dump_helper_custom import dump_results
import pc_util

parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1500, help='Point Number [default: 20000]')
parser.add_argument('--num_target', type=int, default=20, help='Proposal number [default: 256]')
parser.add_argument('--vote_factor', type=int, default=1, help='Vote factor [default: 1]')
parser.add_argument('--cluster_sampling', default='vote_fps', help='Sampling strategy for vote clusters: vote_fps, seed_fps, random [default: vote_fps]')
parser.add_argument('--checkpoint_path', default=None, help='Model checkpoint path [default: None]')
FLAGS = parser.parse_args()

LOG_DIR = ROOT_DIR + '/log_focus'
DEFAULT_CHECKPOINT_PATH = os.path.join(LOG_DIR, 'checkpoint.tar')
CHECKPOINT_PATH = FLAGS.checkpoint_path if FLAGS.checkpoint_path is not None \
	else DEFAULT_CHECKPOINT_PATH

num_input_channel = 1


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net = FocusNet(num_proposal=FLAGS.num_target,
			   input_feature_dim=num_input_channel,
			   vote_factor=FLAGS.vote_factor,
			   sampling=FLAGS.cluster_sampling,testing=True)

net.to(device).eval()
checkpoint = torch.load(CHECKPOINT_PATH)
net.load_state_dict(checkpoint['model_state_dict'])
print('loaded checkpoint:',CHECKPOINT_PATH)


angle_list = [0.0, 0.5235987755982988, 1.0471975511965976, 1.5707963267948966, -1.0471975511965979, -0.5235987755982987]

sigm = nn.Sigmoid()

def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# crop_radius = 50 # pixel units for 0.1m

def visulize_as_image(points,colors,fname):
	img_arr = 255*np.ones((h,w,3))
	P = points_to_pixels_projection(points)
	for i,p in enumerate(P):
		x = int(p[0])
		y = int(p[1])
		img_arr[y,x,:] = colors[i]
	cv2.imwrite(fname,img_arr)

# def get_an_instance_faster(point_cloud,centroid,angle,r,w,h,mask):
# 	x = centroid[0]
# 	y = centroid[1]

# 	xmin = x-r
# 	if xmin < 0:
# 		xmin = 0
# 	xmax = x + r
# 	if xmax > w-1:
# 		xmax = w-1
# 	ymin = y-r
# 	if ymin < 0:
# 		ymin = 0
# 	ymax = y + r
# 	if ymax > h-1:
# 		ymax = h-1

# 	crop = point_cloud[ymin:ymax,xmin:xmax,:]
# 	mask_crop = mask[ymin:ymax,xmin:xmax]
# 	# mask_crop = np.repeat(mask_crop[:, :, np.newaxis], 3, axis=2).astype(np.int32)
# 	crop = crop[np.where(mask_crop==1)]
# 	# visulize_as_image(crop,np.array(crop_color),fname='temp1.png')
# 	# rotate to align the pose horizontally
# 	# angle = -angle_list[j]
# 	# print('angle',angle)
# 	rot_mat = rotz(angle)
# 	crop = np.dot(crop, np.transpose(rot_mat))
# 	# visulize_as_image(crop,np.array(crop_color),fname='temp2.png')
# 	return crop

crop_radius = 0.133
def get_an_instance(point_cloud,angle,centroid):
	crop = []
	for point in point_cloud:
		if np.linalg.norm(point[0:3] - centroid[0:3]) <= crop_radius:
			crop.append(point[0:3])
	crop = np.array(crop)

	# rotate to align the pose horizontally
	# angle = angle_list[j]
	rot_mat = rotz(angle)
	crop = np.dot(crop, np.transpose(rot_mat))
	return crop

def check_for_validity_by_cnn(inputs,centroid,angle):
	# point_cloud = inputs['pc_complete']
	# r = inputs['param'].crop_radius_pixels
	# w = inputs['param'].w
	# h = inputs['param'].h
	# mask = inputs['filter_mask']
	# focused_points = get_an_instance_faster(point_cloud,centroid,angle,r,w,h,mask)

	point_cloud = inputs['pc_cloud']
	# point_cloud = inputs['point_clouds']
	focused_points = get_an_instance(point_cloud,angle,centroid)
	
	focused_points = preprocess_point_cloud(focused_points)
	num_points = FLAGS.num_point
	focused_points, choices = pc_util.random_sampling(focused_points, num_points, return_choices=True)
	pc = np.expand_dims(focused_points.astype(np.float32), 0)
	# Forward pass
	inputs = {'point_clouds': torch.from_numpy(pc).to(device)}
	st = time.time()
	with torch.no_grad():
		end_points = net(inputs)
	preds = sigm(end_points['preds_focus'])[:,0]
	print('CNN time:','**************** ',time.time()-st,' ********************',preds)
	
	if preds > 0.3:
		return True
	else:
		return False

def preprocess_point_cloud(point_cloud):
	''' Prepare the numpy point cloud (N,3) for forward pass '''
	point_cloud = point_cloud[:,0:3] # do not use color for now
	floor_height = np.percentile(point_cloud[:,2],0.99)
	height = point_cloud[:,2] - floor_height
	point_cloud = np.concatenate([point_cloud, np.expand_dims(height, 1)],1) # (N,4) or (N,7)
	# point_cloud = random_sampling(point_cloud, FLAGS.num_point)
	# pc = np.expand_dims(point_cloud.astype(np.float32), 0) # (1,40000,4)
	return point_cloud