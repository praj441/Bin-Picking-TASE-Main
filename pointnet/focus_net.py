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

class FocusNet(nn.Module):
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


		#regression for the per seed point graspability
		self.rgs_net = RgsModule(k=4)
		
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
				
		# -----------------
		xyz = end_points['fp2_xyz']
		features = end_points['fp2_features']
		end_points['seed_inds'] = end_points['fp2_inds']
		end_points['seed_xyz'] = xyz
		end_points['seed_features'] = features
		
		# ---------- Graspability ---------------

		end_points['preds_focus'] = self.rgs_net(end_points['seed_features'])

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

	
