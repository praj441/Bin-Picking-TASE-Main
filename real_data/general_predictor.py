import torch
import numpy as np
import copy
import cv2
import time
import sys
sys.path.append('../commons/')
from custom_grasp_planning_algorithm_dense import select_a_best_grasp_pose
from utils_cnn import draw_final_pose

class Predictor:
	def __init__(self,sampler,data_type,Nt=10):
		self.sampler = sampler
		self.data_type = data_type
		self.Nt = Nt
		if sampler == 'fcn_rgb' or sampler == 'fcn_depth' or sampler == 'fcn_rgbd':
			#initials for fcn backbones
			import torchvision
			import torchvision.transforms as tf
			device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			
			Net = torchvision.models.segmentation.fcn_resnet101(pretrained=True) # Load net
			Net.classifier[4] = torch.nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1)) # Change final layer to 3 classes
			if sampler == 'fcn_rgbd':
				Net.backbone.conv1 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
				self.transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406,0.4), (0.229, 0.224, 0.225,0.2))])
			else:
				self.transformImg=tf.Compose([tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

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

			self.max_pool = torch.nn.MaxPool2d(9, padding=0, stride=9,return_indices=True)

			self.Net = Net


	def predict(self, inputs_np):
		point_cloud_whole = inputs_np['pc_cloud']
		dump_dir = inputs_np['dump_dir']
		sampler = self.sampler

		st = time.time()
		if sampler == 'fcn_rgb' or sampler == 'fcn_depth' or sampler == 'fcn_rgbd':
			if sampler == 'fcn_depth':
				darray = inputs_np['darray']
				darray = darray - darray.min()
				darray = darray/darray.max()
				# print(darray.shape)
				darray = np.tile(darray[:,:,np.newaxis],(1,1,3))
				img = torch.from_numpy((np.transpose(darray, (2, 0, 1)))).float()
			elif sampler == 'fcn_rgb':
				image = imread(sample_path+'_ref_image.png')
				img = torch.from_numpy((np.transpose(image, (2, 0, 1)))).float()
			else:
				img_file = sample_path+'_ref_image.png'
				img = imread(img_file)
				darray = np.loadtxt(sample_path+'_depth_array.txt')
				darray = darray - darray.min()
				darray = darray/darray.max()

				mix_img = np.zeros((img.shape[0],img.shape[1],4))
				mix_img[:,:,0:3] = img[:,:,0:3]
				mix_img[:,:,3] = darray
				img = torch.from_numpy((np.transpose(mix_img, (2, 0, 1)))).float()

			img = self.transformImg(img).unsqueeze(0)
			with torch.no_grad():
				Pred=self.Net(img)['out'].squeeze(0)
			
			# distance sampling
			Pred = Pred.squeeze(0).cpu().numpy()
			Pred_fs = copy.deepcopy(Pred)
			# cv2.imwrite(dump_dir+'/pred.png',Pred)
			Pred = Pred[::8,::8]
			topN_indices = []
			topN_values = []
			dist_thrs = Pred.shape[1]/30
			while len(topN_indices) < self.Nt:
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

			# avg_time += time.time() - st
			# continue
			# print('top_grasp_points',top_grasp_points)
			# print('gqs_score',gqs_score)

		print('grasp sample time',time.time()-st)

		inputs_np['top_grasp_points'] = top_grasp_points
		# inputs_np['angles'] = angles
		inputs_np['final_attempt'] = True
		# inputs_np['param'] = param
		inputs_np['gqs_score'] = gqs_score
		inputs_np['num_dirs'] = 6
		
		st = time.time()
		grasp_pose_info = select_a_best_grasp_pose(inputs_np)
		grasp_pose_info['top_grasp_points'] = top_grasp_points
		grasp_pose_info['Pred'] = Pred_fs
		print('grasp eval time',time.time()-st)
		
		# if dump_dir is not None:
		# 	cv2.imwrite(dump_dir+'/4_final_image.png',final_image)
	
		return grasp_pose_info

		# return outputs

