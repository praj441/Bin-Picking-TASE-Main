import os
import os.path as osp
import numpy as np
import torch
from torch.utils import data
import time
from matplotlib.pyplot import imread
# from scipy.misc import imsave 
from skimage.transform import resize
import cv2 

class Dataset(data.Dataset):
	def __init__(self,path,transform=None,img_size=(240,320),type='rgb_only'):
		self.type = type
		self.path = path
		self.transform = transform
		self.size2D = img_size 

		#read poses and get image names

		#counting number o files
		self.N = len([n for n in os.listdir(osp.join(path, '.')) if
					   n.find('ref_image') >= 0])
		print(self.N)


		

	def __getitem__(self, index):

		if self.type == 'rgb_only':
			img_file = osp.join(self.path,'{0:06d}_ref_image.png'.format(index))
			# print(imread(img_file).shape)
			img = resize(imread(img_file),self.size2D)		
			img = torch.from_numpy((np.transpose(img, (2, 0, 1)))).float()
			img = self.transform(img)

		elif self.type == 'depth_only':

			darray = np.loadtxt(osp.join(self.path,'{0:06d}_depth_array.txt'.format(index)))
			darray = darray - darray.min()
			darray = darray/darray.max()
			# print(darray.shape)
			darray = np.tile(darray[:,:,np.newaxis],(1,1,3))
			img = torch.from_numpy((np.transpose(darray, (2, 0, 1)))).float()
			img = self.transform(img)

		else:
			img_file = osp.join(self.path,'{0:06d}_ref_image.png'.format(index))
			img = resize(imread(img_file),self.size2D)
			darray = np.loadtxt(osp.join(self.path,'{0:06d}_depth_array.txt'.format(index)))
			darray = darray - darray.min()
			darray = darray/darray.max()

			mix_img = np.zeros((img.shape[0],img.shape[1],4))
			mix_img[:,:,0:3] = img[:,:,0:3]
			mix_img[:,:,3] = darray
			img = torch.from_numpy((np.transpose(mix_img, (2, 0, 1)))).float()
			img = self.transform(img)
		# print(darray.shape)
		# print(index)
		# label = np.loadtxt(osp.join(self.path,'{0:06d}_gqs_array.txt'.format(index)))
		label = np.genfromtxt(osp.join(self.path,'{0:06d}_gqs_array.txt'.format(index)))
		# label = np.where((label>=0) & (label<20),0,label)
		# label = np.where((label>=20) & (label<40),1,label)
		# label = np.where((label>=40) & (label<50),2,label)
		# label = np.where((label>=50) & (label<60),3,label)
		# label = np.where((label>=60) & (label<70),4,label)
		# label = np.where((label>=70),5,label)
		label = torch.from_numpy(label).float()

		# raw_img = cv2.imread(osp.join(self.path,'{0:06d}_gqs_image.png'.format(index)))
		return img,label #, raw_img

	def __len__(self):
		return self.N

		

