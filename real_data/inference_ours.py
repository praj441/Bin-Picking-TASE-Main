import numpy as np
import cv2
import time
import sys, os
from datetime import datetime
from math import *
from general_predictor import Predictor
import open3d as o3d
sys.path.append('../commons')
from utils_gs import Parameters
from grasp_evaluation import calculate_GDI2
import copy
from PIL import Image

data_path = 'test_sample_graspnet'
out_path = data_path + '/ours' #'test_data_level_1/ours'
w = 1280 #320
h = 720  #240
param = Parameters(w,h)
param.THRESHOLD2 = 0.005
# self.raw_data_path = os.path.join(ROOT_DIR, 'sunrgbd/sunrgbd_trainval')
scan_names = sorted(list(set([os.path.basename(x)[0:6] \
	for x in os.listdir(data_path)])))

sampler = 'fcn_depth'
data_type = 'mid'
predictor = Predictor(sampler,data_type)

gqcnn_score_list = []
# scan_names = ['000023']
inputs_np = {}
inputs_np['slanted_pose_detection'] = False #False
inputs_np['cone_detection'] = False #False

def process_dmap_for_vis(dmap):
	dmap[dmap<0.4] = dmap.max()
	md = np.loadtxt('20/median_depth_map.txt')
	md = cv2.resize(md,(w,h))
	obj_region = ((md - dmap) > 0.01)
	dmap = np.where(obj_region,dmap-0.4,dmap)
	dv = (dmap/dmap.max()*255)

	obj_region = obj_region 

	dv = np.where(obj_region,np.power(dv,1.5),dv)
	omax = dv[obj_region].max()
	omin = dv[obj_region].min()
	dv = np.where(obj_region,((dv-omin)/omax*255),dv)
	return dv

# scan_names = ['000008']
avg_time = 0.0
scenes = 1
for idx in scan_names:
	print(idx)
	dump_dir = out_path + '/' + idx


	sample_path = os.path.join(data_path, idx)
	# dmap = np.loadtxt(os.path.join(data_path, idx)+'_depth_array.txt')
	dmap = np.array(Image.open(os.path.join(data_path, idx)+'_depth_image.png'))/1000.0
	# dv = process_dmap_for_vis(copy.deepcopy(dmap))
	dv = None
	img = cv2.imread(os.path.join(data_path, idx)+'_ref_image.png')
	# pc_cloud = np.load(os.path.join(data_path, idx)+'_pc.npy')
	# filter_mask = np.load(os.path.join(data_path, idx)+'_filter_mask.npy')
	# pc_complete = np.load(os.path.join(data_path, idx)+'_pc_complete.npy')
	# pcd = o3d.geometry.create_point_cloud_from_depth_image(o3d.geometry.Image(dmap.astype(np.float32)),o3d.camera.PinholeCameraIntrinsic(640,480,614.72,614.14,320.0,240.0),depth_scale=1.0)		
	# 	# o3d.visualization.draw_geometries([pcd])
	# pc_arr = np.asarray(pcd.points).reshape(h,w,3)

	st = time.time()

	inputs_np['image'] = img
	inputs_np['darray'] = dmap
	inputs_np['depth_image'] = dv
	inputs_np['dump_dir'] = dump_dir
	inputs_np['pc_cloud'] = None #pc_cloud
	inputs_np['param'] = param
	# inputs_np['filter_mask'] = filter_mask
	# inputs_np['pc_complete'] = pc_complete
	# inputs_np['pc_arr'] = pc_arr

	final_image,gqs_score = predictor.predict(inputs_np)
	print('time in 1 sample:',time.time()-st)
	# c = input('press 1 and enter to continue')
	avg_time += time.time()-st

	gqcnn_score_list.append(gqs_score)
	
	np.savetxt(out_path+'/score_list.txt',gqcnn_score_list,fmt='%3d')
	cv2.imwrite(out_path+'/final_image_{0}.png'.format(idx),final_image)
	# cv2.imwrite(out_path+'/cluster_image_{0}.png'.format(idx),cluster_image)
	cv2.imwrite(out_path+'/depth_image_{0}.png'.format(idx),dv)
	# cv2.imwrite(out_path+'/image_cnn_pose_{0}.png'.format(idx),image_cnn_pose)
	# c = input('ruko. analyze karo.')
	print('avg_time',avg_time/scenes)
	print('acc',np.count_nonzero(gqcnn_score_list)/scenes)
	scenes += 1
avg_time = avg_time/len(scan_names)

	
	# dv[dv < 150] = 255
	# obj_region = (dv > 150) 
	# dv[dv > 100] = 255
	

	