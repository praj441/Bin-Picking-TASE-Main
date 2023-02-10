# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

""" Deep hough voting network for 3D object detection in point clouds.

Author: Charles R. Qi and Or Litany
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
from backbone_module import Pointnet2Backbone
from voting_module import VotingModule
from cluster_module import ClusterModule
from dump_helper import dump_results
from loss_helper import get_loss
from rgs_module import RgsModule
from graspability_module import GraspabilityModule
from angle_bin_module import AngleBinModule

import torchvision.models.segmentation


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# from loss_helper import map_to_pixels

def map_to_pixels(points,w=320.0,h=240.0,fx=307.36,fy=307.07):
	X = points[:,:,0]
	Y = points[:,:,1]
	Z = points[:,:,2]
	
	# print(points.shape)
	PX = (torch.div(X,Z)*fx + w/2)
	PY = (torch.div(Y,Z)*fx + h/2)

	PX = torch.where(PX < w, PX, torch.tensor(w-1).to(device))
	PX = torch.where(PX >= 0, PX, torch.tensor(0.0).to(device))
	PY = torch.where(PY < h, PY, torch.tensor(h-1).to(device))
	PY = torch.where(PX >= 0, PY, torch.tensor(0.0).to(device))

	PX = (PX/8).to(torch.long)
	PY = (PY/8).to(torch.long)

	P = torch.zeros((PY.shape[0],PY.shape[1]), dtype=torch.long)
	P = PY*40+PX

	return P

class ClusterNet(nn.Module):
	r"""		
		----------
		num_class: int
			Number of semantics classes to predict over -- size of softmax classifier
		num_heading_bin: int
		num_size_cluster: int
		input_feature_dim: (default: 0)
			Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
			value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
		num_proposal: int (default: 128)
			Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
		vote_factor: (default: 1)
			Number of votes generated from each seed point.
	"""

	def __init__(self,input_feature_dim=0, num_proposal=128, vote_factor=1, sampling='seed_fps', testing=False):
		super().__init__()

		# self.num_class = num_class
		# self.num_heading_bin = num_heading_bin
		# self.num_size_cluster = num_size_cluster
		# self.mean_size_arr = mean_size_arr
		# assert(mean_size_arr.shape[0] == self.num_size_cluster)
		self.input_feature_dim = input_feature_dim
		self.num_proposal = num_proposal
		self.vote_factor = vote_factor
		self.sampling=sampling
		self.testing = testing
		# Backbone point feature learning
		self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)
		feature_dim = 768
		# Hough voting
		self.vgen = VotingModule(self.vote_factor, feature_dim)

		# Vote aggregation and detection
		# self.cnet = ClusterModule(num_proposal, sampling)

		#regression for the global number of object prediction
		self.rgs = RgsModule()


		#regression for the per seed point graspability
		self.gqs_net = GraspabilityModule(self.vote_factor, feature_dim)
		self.angle_net = GraspabilityModule(self.vote_factor, feature_dim)
		self.width_net = GraspabilityModule(self.vote_factor, feature_dim)

		self.ab_net = AngleBinModule(self.vote_factor, feature_dim)

		self.cnn = torchvision.models.segmentation.fcn_resnet101(pretrained=True) # Load net
		self.cnn.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # 

		self.activation = {}

		h1 = self.cnn.classifier[0].register_forward_hook(self.getActivation('features'))

	def getActivation(self,name):
	  # the hook signature
	  def hook(model, input, output):
	    self.activation[name] = output #.detach()
	  return hook

	def load_my_state_dict(self, state_dict):
	 
		own_state = self.state_dict()
		for name, param in state_dict.items():
			#print(name)
			if name not in own_state:
				 continue
			print(name,own_state[name].shape)
			print(param.data.shape)
			own_state[name].copy_(param.data)

	def forward(self, inputs):
		""" Forward pass of the network

		Args:
			inputs: dict
				{point_clouds}

				point_clouds: Variable(torch.cuda.FloatTensor)
					(B, N, 3 + input_channels) tensor
					Point cloud to run predicts on
					Each point in the point-cloud MUST
					be formated as (x, y, z, features...)
		Returns:
			end_points: dict
		"""
		end_points = {}
		batch_size = inputs['point_clouds'].shape[0]

		end_points = self.backbone_net(inputs['point_clouds'], end_points)

		cnn_out = self.cnn(inputs['map_2d'])
		cnn_features = self.activation['features']
		cnn_features = cnn_features.view(cnn_features.shape[0],cnn_features.shape[1],-1)



				
		# --------- HOUGH VOTING ---------
		xyz = end_points['fp2_xyz']
		features = end_points['fp2_features']
		end_points['seed_inds'] = end_points['fp2_inds']
		end_points['seed_xyz'] = xyz
		


		end_points['seed_pixels'] = map_to_pixels(end_points['seed_xyz']).unsqueeze(1).repeat(1,cnn_features.shape[1],1)
		# end_points['seed_pixels'] = (end_points['seed_pixels']/8).type(torch.long)
		# cnn_features_mapped = cnn_features[end_points['seed_pixels']]
		# print('cnn_features',cnn_features.shape)
		# print('seed_pixels',end_points['seed_pixels'].shape)
		cnn_features_mapped = torch.gather(cnn_features,2,end_points['seed_pixels'].to(torch.long))
		features = torch.cat((features,cnn_features_mapped),dim=1)

		end_points['seed_features'] = features
		# print('features',features.shape)

		xyz, features = self.vgen(xyz, features)
		features_norm = torch.norm(features, p=2, dim=1)
		features = features.div(features_norm.unsqueeze(1))
		end_points['vote_xyz'] = xyz
		end_points['vote_features'] = features

		num_obj_prediction = self.rgs(features)
		end_points['num_obj_prediction'] = num_obj_prediction
		# print('features',features.shape)
		if self.testing:
			network_predicted_clusters = int(100*num_obj_prediction.detach().cpu().numpy())
			if network_predicted_clusters <= 0:
				network_predicted_clusters = 1
			print('network_predicted_clusters',network_predicted_clusters)
		else:
			network_predicted_clusters = None

		# end_points = self.cnet(xyz, features, end_points,testing=self.testing,num_proposal=network_predicted_clusters)
		
		# ---------- Graspability ---------------

		gqs_score, features = self.gqs_net(end_points['seed_xyz'], end_points['seed_features'])
		features_norm = torch.norm(features, p=2, dim=1)
		features = features.div(features_norm.unsqueeze(1))
		end_points['gqs_score'] = gqs_score
		end_points['gqs_features'] = features

		# angle_score, features = self.angle_net(end_points['seed_xyz'], end_points['seed_features'])
		# features_norm = torch.norm(features, p=2, dim=1)
		# features = features.div(features_norm.unsqueeze(1))
		# end_points['angle_score'] = angle_score
		# end_points['angle_features'] = features

		width_score, features = self.width_net(end_points['seed_xyz'], end_points['seed_features'])
		features_norm = torch.norm(features, p=2, dim=1)
		features = features.div(features_norm.unsqueeze(1))
		end_points['width_score'] = width_score
		end_points['width_features'] = features

		
		ab_score, features = self.ab_net(end_points['seed_xyz'], end_points['seed_features'])
		features_norm = torch.norm(features, p=2, dim=1)
		features = features.div(features_norm.unsqueeze(1))
		end_points['ab_score'] = ab_score
		end_points['ab_features'] = features

		# print(ab_score.shape,features.shape)
		return end_points


if __name__=='__main__':
	sys.path.append(os.path.join(ROOT_DIR, 'sunrgbd'))
	from sunrgbd_detection_dataset import SunrgbdDetectionVotesDataset, DC
	from loss_helper import get_loss

	# Define model
	model = ClusterModule(10,12,10,np.random.random((10,3))).cuda()
	
	try:
		# Define dataset
		TRAIN_DATASET = SunrgbdDetectionVotesDataset('train', num_points=20000, use_v1=True)

		# Model forward pass
		sample = TRAIN_DATASET[5]
		inputs = {'point_clouds': torch.from_numpy(sample['point_clouds']).unsqueeze(0).cuda()}
	except:
		print('Dataset has not been prepared. Use a random sample.')
		inputs = {'point_clouds': torch.rand((20000,3)).unsqueeze(0).cuda()}

	end_points = model(inputs)
	for key in end_points:
		print(key, end_points[key])

	
